from typing import * # pyright: ignore[reportWildcardImportFromLibrary]
from abc import ABC, abstractmethod
from hydra_zen import make_custom_builds_fn

import torch
from torch.nn import functional as F
from torch import nn

from .tasks import TaskSet

class MTLLoss(ABC, nn.Module):
  name: ClassVar[str]

  def __init__(self, dwa_t, tasks):
    ABC.__init__(self)
    nn.Module.__init__(self)

    self.dwa_t = dwa_t
    self.tasks = tasks

    self.losses_dim = sum(task.loss_dim for task in tasks.iter_tasks())
    self.loss_importances = torch.cat([
      torch.full((task.loss_dim,), task.importance, dtype=torch.float32, device="cuda")
      for task in tasks.iter_tasks()
    ], dim=0)
    self.loss_importances /= torch.mean(self.loss_importances)
    self.after_init()

  def after_init(self):
    pass

  def forward(self, losses, external_weights, split):
    # ensure that order is consistent
    losses = [losses[name] for name in self.tasks.active]
    losses = torch.cat([torch.atleast_1d(loss) for loss in losses], dim=0)
    external_weights *= self.loss_importances

    return self.compute_loss(losses, external_weights, split)

  @abstractmethod
  def compute_loss(self, losses, external_weights, split):
    raise NotImplementedError

class UWLoss(MTLLoss):
  name = "uw"
  
  def after_init(self): 
    self.sigma = nn.Parameter(torch.zeros(self.losses_dim))

  # cannot distinguish between internal and external weights
  # todo modulate gradient?
  def compute_loss(self, losses, external_weights, split):
    weights = self.sigma.exp()
    return 0.5 * (losses / weights ** 2).sum() + weights.prod().log()

class DWALoss(MTLLoss):
  name = "dwa"

  def after_init(self):
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

    weights = F.softmax(ratios / self.dwa_t, dim=0)
    return losses @ (weights * external_weights)
    
class RWLoss(MTLLoss):
  name = "rw"

  def compute_loss(self, losses, external_weights, split):
    weights = F.softmax(torch.rand(self.losses_dim, device="cuda"), dim=0)
    return losses @ (weights * external_weights)
  
_constructors_list = [UWLoss, DWALoss, RWLoss]
_constructors_dict = {x.name: x for x in _constructors_list}
# TODO do something similar to taskset with individual configs? idk
def construct_mtl_loss(method, dwa_t, tasks) -> MTLLoss:
  constructor = _constructors_dict.get(method)
  if constructor is None:
    raise ValueError(f"invalid MTL method: {method}")

  return constructor(dwa_t, tasks)

builds = make_custom_builds_fn(populate_full_signature=True)

MTLLossCfg = builds(
  construct_mtl_loss,
  method = "rw",
  dwa_t = 2.0,
  zen_partial = True
)
class PartialMTLLoss(Protocol):
  def __call__(self, tasks: TaskSet) -> MTLLoss: ...
