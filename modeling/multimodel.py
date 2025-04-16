from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

from lightning import LightningModule

class TaskSetLoss(nn.Module):
  def __init__(self, config, tasks):
    super().__init__()
    self.config = config
    self.losses_dim = sum(task.loss_dim for task in tasks.values())
    self.loss_importances = torch.Tensor(np.concatenate([
      np.full(task.loss_dim, task.importance) for task in tasks.values()
    ], axis=0)).cuda()
    self.loss_importances *= self.losses_dim / torch.sum(self.loss_importances)
    self.loss_norm = None 
    self.norm_loss_history = []
    self.dwa_loss_history = []

  def forward(self, losses, split):
    losses_flat = torch.cat([torch.atleast_1d(loss) for loss in losses], dim=0)
    losses_norm = losses_flat * (1 if self.loss_norm is None else self.loss_norm)
          
    # norm
    if split == "train" and self.config["mtl_norm_initial"]:
      cap_diff = self.config["mtl_norm_length"] - len(self.norm_loss_history)
      if cap_diff >= 1:
        self.norm_loss_history.append(losses_norm.detach())
      if cap_diff == 1:
        mean_losses = torch.mean(torch.stack(self.norm_loss_history), dim=0) 
        self.loss_norm = 1 / (mean_losses + 1e-8)

    # DWA
    if len(self.dwa_loss_history) == 2:
      ratios = self.dwa_loss_history[1] / (self.dwa_loss_history[0] + 1e-8)
    else:
      ratios = torch.ones(self.losses_dim).cuda()

    if split == "train":
      if len(self.dwa_loss_history) == 2:
        self.dwa_loss_history.pop(0)
      self.dwa_loss_history.append(losses_norm.detach())

    softmax = F.softmax(ratios / self.config["mtl_dwa_T"], dim=0)
    return losses_norm @ (softmax * self.loss_importances)
  
class BaseMultiModel(ABC, LightningModule):
  def __init__(self, config, task_sets):
    ABC.__init__(self)
    LightningModule.__init__(self)

    # make cuda TODO check if can just do in BaseTask
    task_sets = {set_name: {
      name: task.cuda() for name, task in task_set.items()
    } for set_name, task_set in task_sets.items()}

    self.config = config
    self.task_sets = task_sets
    self.set_losses = {
      set_name: TaskSetLoss(config, task_set)
      for set_name, task_set in task_sets.items()
    }
    self.set_names = list(task_sets.keys())

  @abstractmethod
  def forward_base(self, batch):
    pass
    
  def split_step(self, named_batches, split):
    batch_sizes, metricsss, lossess = {}, {}, {}
    for set_name, set_batch in named_batches.items():
      hidden = self.forward_base(set_batch)
      batch_sizes[set_name] = next(iter(set_batch.values())).shape[0]

      metricss, losses = {}, {}
      for task_name, task in self.task_sets[set_name].items():
        metricss[task_name], losses[task_name] = task.compute(hidden, set_batch, split)
      metricsss[set_name], lossess[set_name] = metricss, losses
    
    log_kwargs = {"prog_bar": True, "on_epoch": True, "on_step": False}

    # log metrics
    if split != "train":
      for set_name, metricss in metricsss.items():
        for metrics in metricss.values():
          self.log_dict(metrics, batch_size=batch_sizes[set_name], **log_kwargs)

    # log losses
    # if split != "test":
    #   for name, loss in losses.items():
    #     if loss.dim == 0:
    #       self.log(f"{split}_{name}_loss", loss, **log_kwargs)
    #     elif loss.dim == 1:
    #       for i, sub_loss in enumerate(loss):
    #         self.log(f"{split}_{name}_loss_{i}", sub_loss, **log_kwargs)

    loss = self.config["batch_size"] * sum(
      self.set_losses[set_name](losses.values(), split) / batch_sizes[set_name] for set_name, losses in lossess.items()
    )
    # TODO log?
    # TODO only when split != test?
    return loss
    
  def training_step(self, batch):
    return self.split_step(batch, "train")
 
  def validation_step(self, batch, _):
    return self.split_step(batch, "valid")
  
  def test_step(self, batch, _):
    return self.split_step(batch, "test")

  def on_split_epoch_end(self, split):
    for task_sets in self.task_sets.values():
      for task in task_sets.values():
        task.metrics[split].reset()

  def on_train_epoch_end(self):
    self.on_split_epoch_end("train")
  
  def on_validation_epoch_end(self):
    self.on_split_epoch_end("valid")
    
  def on_test_epoch_end(self):
    self.on_split_epoch_end("test")

