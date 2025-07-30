from typing import *
from pydantic import BaseModel

import numpy as np
import csv
import torch

from pytorch_optimizer import load_optimizer, Lookahead
from torch.optim.lr_scheduler import *
from transformers import ElectraModel, BitsAndBytesConfig, QuantoConfig, AutoModel
from lightning import LightningModule

from .methods import AdapterMethod
from .heads import HateHeads
from .mtl_loss import MTLLoss
from .tasks import TaskSet
from .misc import MultiLR
from common import fbuilds, store

class HateOptimization(BaseModel):
  name: str
  learning_rate: float
  weight_decay: float
  use_lookahead: bool
  warmup: int

  def build_optimizer(self, model):
    optimizer_func = load_optimizer(self.name)
    params = [
      {"params": model.heads.parameters()},
      {"params": model.encoder.parameters()},
    ]
    optimizer = optimizer_func(
      params=params,
      lr=self.learning_rate, # pyright: ignore
      weight_decay= self.weight_decay, # pyright: ignore
    )

    if self.use_lookahead:
      optimizer = Lookahead(optimizer)

    return optimizer
  
  def build_head_scheduler(self, optimizer):
    return CosineAnnealingLR(
      optimizer,
      T_max=1,
      eta_min=0.01 * self.learning_rate
    )

  def build_encoder_scheduler(self, optimizer):
    return SequentialLR(
      optimizer,
      schedulers=[
        LinearLR(
          optimizer,
          start_factor=0.01,
          end_factor=1.0,
          total_iters=self.warmup,
        ), CosineAnnealingLR(
          optimizer,
          T_max=1,
          eta_min=0.01*self.learning_rate,
        )
      ],
      milestones=[self.warmup],
    )
  
  def build(self, model):
    optimizer = self.build_optimizer(model)
    scheduler = MultiLR(
      optimizer,
      [self.build_head_scheduler, self.build_encoder_scheduler]
    )
    scheduler_dict = {"scheduler": scheduler, "interval": "epoch"}
    return [optimizer], [scheduler_dict]

class HateModule(LightningModule):
  def __init__(
    self,
    model: ElectraModel,
    method: AdapterMethod,
    heads: HateHeads,
    optimization: HateOptimization,
    tasks: TaskSet,
    mtl_loss: MTLLoss,
    save_path: str,
  ):
    super().__init__()
    # Self.save_hyperparameters()

    self.encoder = method.apply(model)
    self.heads = heads
    self.optimization = optimization
    self.tasks = tasks
    self.mtl_loss = mtl_loss
    self.save_path = save_path
    self.results = None

  def vis_params(self):
    params = list(self.encoder.named_parameters())
    max_name_len = max(len(k) for k,_ in params)
    grad_params = [(k, v) for k,v in params if v.requires_grad]
    no_grad_params = [(k, v) for k,v in params if not v.requires_grad]
    print(f"grad params:")
    for k, v in grad_params:
      print(f"  {k:{max_name_len}}  {str(v.dtype):15s}  {str(v.shape):15s}")
    print(f"no grad params:")
    for k, v in no_grad_params:
      print(f"  {k:{max_name_len}}  {str(v.dtype):15s}  {str(v.shape):15s}")
  
  def _write_results(self, path):
    if self.results is not None:
      writer = csv.DictWriter(open(path, "w"), self.results.keys())
      writer.writeheader()
      writer.writerow(self.results)
  
  def save(self, action: str):
    if self.save_path is None:
      print("not saving, no save_path specified")
    elif action == "train":
      print(f"saving to {self.save_path}")
      self.encoder.save_all_adapters(f"{self.save_path}/encoder")
      self.heads.save(f"{self.save_path}/heads.pt")
      self._write_results(f"{self.save_path}/results.csv")
    elif action == "test":
      self._write_results(f"{self.save_path}/results.csv")

  def forward_base(self, batch):
    return self.encoder(
      input_ids=batch["tokens"],
      attention_mask=batch["mask"].bfloat16()
    )

  def forward(self, batches, split):
    log_metricss, losses, sizes = {}, {}, {}
    for set_name in self.tasks.datasets():
      batch = batches[set_name]
      output = self.forward_base(batch)

      for task_name, task in self.tasks.items():
        if task.dataset == set_name:
          log_metricss[task_name], losses[task_name] = self.heads[task_name].compute(output, batch, split)
          sizes[task_name] = batch["size"]

    return log_metricss, losses, sizes

  def loss_weights(self, sizes):
    loss_counts = torch.Tensor(np.concatenate([
      np.full(task.mask_sum, sizes[name])
      for name, task in self.tasks.items()
    ], axis=0)).to(self.encoder.device)
    return loss_counts / loss_counts.mean()

  def configure_optimizers(self): # pyright: ignore
    return self.optimization.build(self)
  
  def split_step(self, batches, split):
    if any(x["size"] == 0 for x in batches.values()):
      return None
    
    log_metricss, losses, sizes = self.forward(batches, split)
    weights = self.loss_weights(sizes)
      
    log_kwargs = {"prog_bar": split == "test", "on_epoch": True, "on_step": False}

    # log metrics
    for name, log_metrics in log_metricss.items():
      log_metrics(self, batch_size=sizes[name], **log_kwargs)
      # self.log_dict(log_metrics, batch_size=sizes[name], **log_kwargs)

    # log losses
    if split != "test":
      for name, loss in losses.items():
        loss = loss.float().cpu().detach().numpy()
        for idx, val in np.ndenumerate(loss):
          str_idx = "_".join(str(x) for x in idx)
          prefix = "_" if str_idx else ""
          log_name = f"{split}_{name}_loss{prefix}{str_idx}"
          self.log(log_name, val, batch_size=sizes[name], **log_kwargs)

    return self.mtl_loss(losses, weights, split)

  def state_dict(self, *args, **kwargs):
    state = super().state_dict(*args, **kwargs)
    quant_ends = ('absmax', 'scale', 'quant_map', 'bitsandbytes__nf4')
    return {k: v for k, v in state.items() if not k.endswith(quant_ends)}

  # def dequantize(self):
  #   self.encoder = self.encoder.to("cuda")
  #   self.encoder.dequantize()
    
  def training_step(self, batch):
    return self.split_step(batch, "train")
 
  def validation_step(self, batch, _):
    return self.split_step(batch, "valid")
  
  def test_step(self, batch, _):
    return self.split_step(batch, "test")

  def on_split_epoch_end(self, split):
    for name in self.tasks.names():
      self.heads[name].metrics[split].reset()

  def on_train_epoch_end(self):
    self.on_split_epoch_end("train")
  
  def on_validation_epoch_end(self):
    self.on_split_epoch_end("valid")
    
  def on_test_epoch_end(self):
    self.results = {
      k: v for head in self.heads.list()
      for k, v in head.metrics["test"].compute().items() # pyright: ignore
    }
    self.on_split_epoch_end("test")

quant_store = store(group="quantization")
quant_store(fbuilds(
  BitsAndBytesConfig,
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=True,
  bnb_4bit_compute_dtype="bfloat16",
), name="nf4-double")
quant_store(fbuilds(
  BitsAndBytesConfig,
  load_in_4bit=True,
  bnb_4bit_quant_type="nf4",
  bnb_4bit_use_double_quant=False,
  bnb_4bit_compute_dtype="bfloat16",
), name="nf4-single")
quant_store(fbuilds(
  BitsAndBytesConfig,
  load_in_8bit=True,
  bnb_4bit_compute_dtype="bfloat16",
), name="8bit")
quant_store(fbuilds(
  QuantoConfig,
  weights="int8",
), name="quanto8")
quant_store(fbuilds(
  QuantoConfig,
  weights="int4",
), name="quanto4")

def ret_none(): return None
quant_store(fbuilds(ret_none), name="none")

model_store = store(group="model")
model_store(fbuilds(
  AutoModel.from_pretrained,
  pretrained_model_name_or_path="google/electra-small-discriminator",
  device_map="auto",
  torch_dtype="bfloat16",
  low_cpu_mem_usage=True,
  quantization_config="${quantization}",
), name="electra-small")
model_store(fbuilds(
  AutoModel.from_pretrained,
  pretrained_model_name_or_path="google/electra-base-discriminator",
  device_map="auto",
  torch_dtype="bfloat16",
  low_cpu_mem_usage=True,
  quantization_config="${quantization}",
), name="electra-base")

optimization_store = store(group="optimization")
levels = {
  "slow": 1e-4,
  "medium": 2.5e-4,
  "fast": 5e-4,
  "faster": 1e-3,
  "fastest": 2e-3,
}
for name, learning_rate in levels.items():
  optimization_store(fbuilds(
    HateOptimization,
    name="adamw",
    learning_rate=learning_rate,
    weight_decay=1e-2,
    use_lookahead=True,
    warmup=2,
  ), name=name)

HateModuleCfg = fbuilds(
  HateModule,
  method="${method}",
  model="${model}",
  optimization="${optimization}",
  heads="${heads}",
  tasks="${tasks}",
  mtl_loss="${mtl_loss}",
  save_path="${save_path}",
)
