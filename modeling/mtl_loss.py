from typing import *
from abc import ABC, abstractmethod

import torch
from torch.nn import functional as F
from torch import nn

from hydra_zen import store

from .tasks import TaskSet
from .utils import *

class MTLLoss(ABC, nn.Module):
  def __init__(self, tasks: TaskSet):
    ABC.__init__(self)
    nn.Module.__init__(self)
    self.tasks = tasks

    self.losses_dim = sum(task.loss_dim for task in tasks.values())
    self.loss_importances = torch.cat([
      torch.full((task.loss_dim,), task.importance, dtype=torch.float32, device="cuda")
      for task in tasks.values()
    ], dim=0)
    self.loss_importances /= torch.mean(self.loss_importances)

  def forward(self, losses, external_weights, split):
    # ensure that order is consistent
    losses = [losses[name] for name in self.tasks.names()]
    losses = torch.cat([torch.atleast_1d(loss) for loss in losses], dim=0)
    external_weights *= self.loss_importances

    return self.compute_loss(losses, external_weights, split)

  @abstractmethod
  def compute_loss(self, losses, external_weights, split):
    raise NotImplementedError

class UWLoss(MTLLoss):
  def __init__(self, tasks: TaskSet):
    super().__init__(tasks)
    self.sigma = nn.Parameter(torch.zeros(self.losses_dim))

  # cannot distinguish between internal and external weights
  # todo modulate gradient?
  def compute_loss(self, losses, external_weights, split):
    weights = self.sigma.exp()
    return 0.5 * (losses / weights ** 2).sum() + weights.prod().log()

class DWALoss(MTLLoss):
  def __init__(self, tasks: TaskSet, t: float):
    super().__init__(tasks)
    self.t = t
    self.history = []

  def compute_loss(self, losses, external_weights, split):
    if len(self.history) == 2:
      ratios = self.history[1] / (self.history[0] + 1e-8)
    else:
      ratios = torch.ones(self.losses_dim).cuda()

    if split == "train":
      self.history.append(losses.detach())
      if len(self.history) > 2:
        self.history.pop(0)

    weights = F.softmax(ratios / self.t, dim=0)
    return losses @ (weights * external_weights)
    
class RWLoss(MTLLoss):
  def compute_loss(self, losses, external_weights, split):
    weights = F.softmax(torch.rand(self.losses_dim, device="cuda"), dim=0)
    return losses @ (weights * external_weights)

mtl_loss_store = store(group="mtl_loss")
mtl_loss_store(fbuilds(UWLoss, tasks="${tasks}"), name="uw")
mtl_loss_store(fbuilds(DWALoss, tasks="${tasks}", t=4.0), name="dwa")
mtl_loss_store(fbuilds(RWLoss, tasks="${tasks}"), name="rw")
