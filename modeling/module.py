from typing import *
from pydantic import BaseModel
from hydra_zen import store

import numpy as np
import torch

from pytorch_optimizer import create_optimizer
from torch.optim.lr_scheduler import *
from transformers import ElectraModel, BitsAndBytesConfig, QuantoConfig, AutoModel

from lightning import LightningModule

from .methods import AdapterMethod
from .heads import HateHeads
from .mtl_loss import MTLLoss
from .tasks import TaskSet
from .utils import *

class HateOptimization(BaseModel):
  name: str
  learning_rate: float
  weight_decay: float
  use_lookahead: bool

  def build(self, model):
    return create_optimizer(
      model,
      self.name,
      lr=self.learning_rate,
      weight_decay=self.weight_decay,
      use_lookahead=self.use_lookahead
    )

class HateModule(LightningModule):
  def __init__(
    self,
    model: ElectraModel,
    method: AdapterMethod,
    heads: HateHeads,
    optimization: HateOptimization,
    tasks: TaskSet,
    mtl_loss: MTLLoss,
    output_path: str | None,
  ):
    super().__init__()
    # Self.save_hyperparameters()

    self.encoder = method.apply(model)
    self.heads = heads
    self.optimization = optimization
    self.tasks = tasks
    self.mtl_loss = mtl_loss
    self.output_path = output_path

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

  def save(self):
    if self.output_path is None:
      print("not saving, no output_path specified")
    else:
      print(f"saving to {self.output_path}")
      self.encoder.save_all_adapters(self.output_path)

  def forward_base(self, batch):
    return self.encoder(
      input_ids=batch["tokens"],
      attention_mask=batch["mask"].bfloat16()
    )

  def forward(self, batches, split):
    metricss, losses, sizes = {}, {}, {}
    for set_name in self.tasks.datasets():
      batch = batches[set_name]
      output = self.forward_base(batch)

      for task_name, task in self.tasks.items():
        if task.dataset == set_name:
          metricss[task_name], losses[task_name] = self.heads[task_name].compute(output, batch, split)
          sizes[task_name] = batch["size"]

    return metricss, losses, sizes

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
    
    metricss, losses, sizes = self.forward(batches, split)
    weights = self.loss_weights(sizes)
      
    log_kwargs = {"prog_bar": split == "test", "on_epoch": True, "on_step": False}

    # log metrics
    for name, metrics in metricss.items():
      self.log_dict(metrics, batch_size=sizes[name], **log_kwargs)

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
    use_lookahead=False,
  ), name=name)

HateModuleCfg = fbuilds(
  HateModule,
  method="${method}",
  model="${model}",
  optimization="${optimization}",
  heads="${heads}",
  tasks="${tasks}",
  mtl_loss="${mtl_loss}",
  output_path=None,
)
