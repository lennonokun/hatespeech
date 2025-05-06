from abc import ABC, abstractmethod

import torch
import numpy as np
from torch import nn

from lightning import LightningModule

from .mtl_loss import construct_mtl_loss
from .tasks import construct_tasks
  
class BaseModel(ABC, LightningModule):
  def __init__(self, config):
    ABC.__init__(self)
    LightningModule.__init__(self)
    self.save_hyperparameters()

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
