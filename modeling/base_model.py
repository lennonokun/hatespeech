from abc import ABC, abstractmethod

import numpy as np
import torch
from torch import nn

from transformers import AutoModel, BitsAndBytesConfig
from bitsandbytes.optim import *
from torch.optim.lr_scheduler import *

from lightning import LightningModule

from .mtl_loss import construct_mtl_loss
from .tasks import construct_tasks
  
class BaseModel(ABC, LightningModule):
  def __init__(self, config, do_quantize):
    ABC.__init__(self)
    LightningModule.__init__(self)
    self.save_hyperparameters()

    bnb_config = BitsAndBytesConfig(
      load_in_4bit=True,
      bnb_4bit_quant_type="nf4",
      bnb_4bit_use_double_quant=True,
      bnb_4bit_compute_dtype=torch.bfloat16,
    )
    self.model = AutoModel.from_pretrained(
      config["model"],
      low_cpu_mem_usage=True,
      device_map="auto",
      torch_dtype=torch.bfloat16,
      quantization_config=bnb_config if do_quantize else None,
    )

    self.lr = config["learning_rate"]
    self.config = config
    self.tasks = nn.ModuleDict(construct_tasks(config))
    self.mtl_loss = construct_mtl_loss(config, self.tasks)
    # SHOULD BE SET IN SUBCLASS __init__ OR NO GRAD NORM
    self.norm_layers = None

  @abstractmethod
  def forward(self, batches, split):
    raise NotImplementedError   
    
  def split_step(self, batches, split):
    if any(x["size"] == 0 for x in batches.values()):
      return None
    
    metricss, losses, sizes, weights = self.forward(batches, split)
      
    log_kwargs = {"prog_bar": split == "test", "on_epoch": True, "on_step": False}

    # log metrics
    for name, metrics in metricss.items():
      self.log_dict(metrics, batch_size=sizes[name], **log_kwargs)

    # log losses
    if split != "test":
      for name, loss in losses.items():
        loss = loss.cpu().detach().numpy()
        for idx, val in np.ndenumerate(loss):
          str_idx = "_".join(str(x) for x in idx)
          prefix = "_" if str_idx else ""
          log_name = f"{split}_{name}_loss{prefix}{str_idx}"
          self.log(log_name, val, batch_size=sizes[name], **log_kwargs)

    # ensure that order is consistent
    losses = [losses[task_name] for task_name in self.config["melt_tasks"]]
    return self.mtl_loss(losses, self.norm_layers, weights, split)

  def configure_optimizers(self): # pyright: ignore
    std_params = filter(lambda p: p.requires_grad, self.parameters())
    optimizer = PagedAdamW32bit(params=std_params, lr=self.lr)
    # optimizer = AdamW(params=std_params, lr=self.lr)
    warmup_scheduler = LinearLR(
      optimizer,
      start_factor=0.01,
      end_factor=1.0,
      total_iters=3
    )
    cosine_scheduler = CosineAnnealingLR(
      optimizer,
      T_max=1,
      eta_min=0.1*self.lr,
    )
    scheduler = SequentialLR(
      optimizer,
      schedulers=[warmup_scheduler, cosine_scheduler],
      milestones=[3]
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

  def state_dict(self, *args, **kwargs):
    state = super().state_dict(*args, **kwargs)
    quant_ends = ('absmax', 'scale', 'quant_map', 'bitsandbytes__nf4')
    return {k: v for k, v in state.items() if not k.endswith(quant_ends)}
    
  def training_step(self, batch):
    return self.split_step(batch, "train")
 
  def validation_step(self, batch, _):
    return self.split_step(batch, "valid")
  
  def test_step(self, batch, _):
    return self.split_step(batch, "test")

  def on_split_epoch_end(self, split):
    for task in self.tasks.values():
      task.metrics[split].reset()
    self.mtl_loss.reset_norms()

  def on_train_epoch_end(self):
    self.on_split_epoch_end("train")
  
  def on_validation_epoch_end(self):
    self.on_split_epoch_end("valid")
    
  def on_test_epoch_end(self):
    self.on_split_epoch_end("test")
