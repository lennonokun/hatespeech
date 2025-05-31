from typing import * # pyright: ignore[reportWildcardImportFromLibrary]
from pydantic import BaseModel
from hydra_zen import builds

import numpy as np
import torch

from transformers import AutoModel, ElectraModel, BitsAndBytesConfig

import adapters
from adapters.models.bert.mixin_bert import BertModelAdaptersMixin
from adapters import AdapterConfig, LoRAConfig

from bitsandbytes.optim import *
from torch.optim.lr_scheduler import *

from lightning import LightningModule

from .heads import HateHeads
from .mtl_loss import MTLLoss
from .tasks import TaskSet
from .custom import MultiLR

class HateOptimization(BaseModel):
  learning_rate: float
  warmup: int

  def build_head_scheduler(self, optimizer):
    return CosineAnnealingLR(
      optimizer,
      T_max=1,
      eta_min=0.05*self.learning_rate,
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

  def build(self, head_params, encoder_params):
    optimizer = PagedAdamW32bit([
      {"params": head_params, "lr": self.learning_rate},
      {"params": encoder_params, "lr": 5 * self.learning_rate}
    ])

    scheduler = MultiLR(optimizer, [self.build_head_scheduler, self.build_encoder_scheduler])
    scheduler_dict = {"scheduler": scheduler, "interval": "epoch"}
    return [optimizer], [scheduler_dict]

class AdapterEncoder(BertModelAdaptersMixin, ElectraModel): # pyright: ignore
  ...

def build_encoder(
  model: ElectraModel,
  adapter_config: AdapterConfig,
) -> AdapterEncoder:
  adapters.init(model)
  model = cast(AdapterEncoder, model)

  model.add_adapter("adapter", adapter_config)
  # model.set_active_adapters(["adapter"])
  model.train_adapter("adapter") # pyright: ignore
  
  for name, param in model.named_parameters():
    if "adapter" in name or "lora" in name:
      param.data = param.data.to(torch.bfloat16)
  
  return model

class HateModule(LightningModule):
  def __init__(
    self,
    encoder: AdapterEncoder,
    heads: HateHeads,
    optimization: HateOptimization,
    tasks: TaskSet,
    mtl_loss: MTLLoss,
  ):
    super().__init__()
    # Self.save_hyperparameters()

    self.encoder = encoder
    self.heads = heads
    self.optimization = optimization
    self.tasks = tasks
    self.mtl_loss = mtl_loss

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

  def adjust_dtypes(self):
    for name, param in self.encoder.named_parameters():
      if "adapter" in name or "lora" in name:
        param.data = param.data.to(torch.bfloat16)

  def save(self, dest):
    if dest is not None:
      self.encoder.save_all_adapters(dest)

  def forward_base(self, batch):
    return self.encoder(
      input_ids=batch["tokens"],
      attention_mask=batch["mask"].bfloat16()
    ).last_hidden_state.float()

  def forward(self, batches, split):
    metricss, losses, sizes = {}, {}, {}
    for set_name in self.tasks.datasets:
      batch = batches[set_name]
      hidden = self.forward_base(batch)

      for task_name, task in self.tasks.iter_pairs():
        if task.dataset == set_name:
          metricss[task_name], losses[task_name] = self.heads[task_name].compute(hidden, batch, split)
          sizes[task_name] = batch["size"]

    return metricss, losses, sizes

  def loss_weights(self, sizes):
    loss_counts = torch.Tensor(np.concatenate([
      np.full(task.loss_dim, sizes[name])
      for name, task in self.tasks.iter_pairs()
    ], axis=0)).to(self.encoder.device)
    return loss_counts / loss_counts.mean()

  def configure_optimizers(self): # pyright: ignore
    head_params = self.heads.parameters()
    encoder_params = [p for p in self.encoder.parameters() if p.requires_grad]
    return self.optimization.build(head_params, encoder_params)
    
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
    for name in self.tasks.active:
      self.heads[name].metrics[split].reset()

  def on_train_epoch_end(self):
    self.on_split_epoch_end("train")
  
  def on_validation_epoch_end(self):
    self.on_split_epoch_end("valid")
    
  def on_test_epoch_end(self):
    self.on_split_epoch_end("test")

OptimizationCfg = builds(
  HateOptimization,
  learning_rate=1e-4,
  warmup=4,
)

BnbCfg = builds(
  BitsAndBytesConfig,
  load_in_4bit = True,
  bnb_4bit_quant_type = "nf4",
  bnb_4bit_use_double_quant = True,
  bnb_4bit_compute_dtype = "bfloat16",
)

# todo model config?
BaseEncoderCfg = builds(
  AutoModel.from_pretrained,
  pretrained_model_name_or_path="google/electra-small-discriminator",
  low_cpu_mem_usage=True,
  device_map="auto",
  torch_dtype="bfloat16",
  quantization_config=BnbCfg,
)

# todo: multiple options
AdapterCfg = builds(
  LoRAConfig,
  r=8,
  alpha=8,
  dropout=0.1,
  dtype="bfloat16",
)

EncoderCfg = builds(
  build_encoder,
  model=BaseEncoderCfg,
  adapter_config=AdapterCfg,
)
