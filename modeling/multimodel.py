from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

from lightning import LightningModule

from .tasks import TargetTask, RationaleTask, LabelTask, ScoreTask

class MTLLoss(nn.Module):
  def __init__(self, config, tasks):
    super().__init__()
    self.config = config
    self.losses_dim = sum(task.loss_dim for task in tasks.values())
    self.loss_importances = torch.Tensor(np.concatenate([
      np.full(task.loss_dim, task.importance) for task in tasks.values()
    ], axis=0)).cuda()
    self.loss_importances *= self.losses_dim / torch.sum(self.loss_importances)

    self.loss_norm = None 
    self.norm_history = []
    self.history_wait = 0

    self.dwa_loss_history = []

  def reset_norms(self):
    self.loss_norm = None
    self.norm_history = []
  
  def compute_norms(self, losses_flat, norm_layers):
    norms = []
    for loss in losses_flat:
      grad = torch.autograd.grad(
        loss, norm_layers, create_graph=True, retain_graph=True
      )[0].detach()
      norms.append(torch.norm(grad))
    return torch.stack(norms)
  
  def forward(self, losses, norm_layers, split):
    losses_flat = torch.cat([torch.atleast_1d(loss) for loss in losses], dim=0)
    losses_norm = losses_flat * (1 if self.loss_norm is None else self.loss_norm)
          
    if split == "train" and self.config["mtl_norm_do"]:
      self.history_wait += 1

      if self.history_wait == self.config["mtl_norm_period"]:
        self.norm_history.append(self.compute_norms(losses_flat, norm_layers))

        if len(self.norm_history) > self.config["mtl_norm_length"]:
          self.norm_history.pop(0)
        
        mean_norms = torch.mean(torch.stack(self.norm_history), dim=0) 
        self.loss_norm = torch.mean(mean_norms) / (mean_norms + 1e-8)
        self.history_wait = 0
    
    # DWA
    if len(self.dwa_loss_history) == 2:
      ratios = self.dwa_loss_history[1] / (self.dwa_loss_history[0] + 1e-8)
    else:
      ratios = torch.ones(self.losses_dim).cuda()

    if split == "train" and self.config["mtl_weighing"] == "dwa":
      self.dwa_loss_history.append(losses_flat.detach())
      if len(self.dwa_loss_history) > 2:
        self.dwa_loss_history.pop(0)

    softmax = F.softmax(ratios / self.config["mtl_dwa_T"], dim=0)
    return losses_norm @ (softmax * self.loss_importances)

    # weights = F.softmax(torch.rand(self.losses_dim, device="cuda"), dim=0)
    # return losses_norm @ (weights * self.loss_importances)
  
class BaseMultiModel(ABC, LightningModule):
  _task_constructors = {
    "target": TargetTask,
    "rationale": RationaleTask,
    "label": LabelTask,
    "score": ScoreTask,
  }

  def __init__(self, config):
    ABC.__init__(self)
    LightningModule.__init__(self)

    self.config = config
    self.tasks = nn.ModuleDict({
      name: self._task_constructors[name](config)
      for name in config["melt_tasks"]
    }).cuda()
    self.mtl_loss = MTLLoss(config, self.tasks)
    self.norm_layers = None
  
  def set_norm_layers(self, value):
    self.norm_layers = value

  @abstractmethod
  def forward_base(self, batch):
    pass
    
  def split_step(self, named_batches, split):
    hiddens = {}
    for set_name in self.config["flat_datasets"]:
      hiddens[set_name] = self.forward_base(named_batches[set_name])
    
    batch_sizes, metricss, losses = {}, {}, {}
    for set_name, task_name in self.config["melt_pairs"]:
      batch = named_batches[set_name]
      task = self.tasks[task_name]
      hidden = hiddens[set_name]

      batch_size = next(iter(batch.values())).shape[0]

      metricss[task_name], losses[task_name] = task.compute(hidden, batch, split)
      batch_sizes[task_name] = batch_size
    
    log_kwargs = {"prog_bar": split == "test", "on_epoch": True, "on_step": False}
    # log metrics
    for name, metrics in metricss.items():
      self.log_dict(metrics, batch_size=batch_sizes[name], **log_kwargs)

    # log losses
    if split != "test":
      for name, loss in losses.items():
        b_size = batch_sizes[name]
        if loss.dim() == 0:
          self.log(f"{split}_{name}_loss", loss, batch_size=b_size, **log_kwargs)
        elif loss.dim() == 1:
          for i, sub_loss in enumerate(loss):
            self.log(f"{split}_{name}_loss_{i}", sub_loss, batch_size=b_size, **log_kwargs)

    # ensure that order is consistent
    losses = [losses[task_name] for task_name in self.config["melt_tasks"]]
    return self.mtl_loss(losses, self.norm_layers, split)
    
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

