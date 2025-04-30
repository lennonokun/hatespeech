from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

class MTLGradNorm(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.weights = None
    self.history = []
    self.wait = 0

  def reset(self):
    self.weights = None
    self.history = []

  def compute_norms(self, losses_flat, norm_layers):
    norms = []
    for loss in losses_flat:
      grad = torch.autograd.grad(
        loss, norm_layers, create_graph=True, retain_graph=True
      )[0].detach()
      norms.append(torch.norm(grad))
    return torch.stack(norms)

  def forward(self, losses, norm_layers, split):
    if split == "train":
      self.wait += 1

      if self.wait == self.config["mtl_norm_period"]:
        self.history.append(self.compute_norms(losses, norm_layers))

        if len(self.history) > self.config["mtl_norm_length"]:
          self.history.pop(0)
        
        mean_norms = torch.mean(torch.stack(self.history), dim=0) 
        self.weights = torch.mean(mean_norms) / (mean_norms + 1e-8)
        self.wait = 0

    if self.weights is not None:
      return self.weights
    else:
      return torch.ones_like(losses)

class MTLLoss(ABC, nn.Module):
  def __init__(self, config, tasks):
    ABC.__init__(self)
    nn.Module.__init__(self)

    self.config = config
    self.losses_dim = sum(task.loss_dim for task in tasks.values())
    self.loss_importances = torch.Tensor(np.concatenate([
      np.full(task.loss_dim, task.importance) for task in tasks.values()
    ], axis=0)).cuda()
    self.loss_importances /= torch.mean(self.loss_importances)
    self.grad_norm = MTLGradNorm(config) if config["mtl_norm_do"] else None

  def forward(self, losses, norm_layers, external_weights, split):
    losses = torch.cat([torch.atleast_1d(loss) for loss in losses], dim=0)
    external_weights *= self.loss_importances
    if self.grad_norm is not None:
      external_weights *= self.grad_norm(losses, norm_layers, split)

    return self.compute_loss(losses, external_weights, split)

  def reset_norms(self):
    if self.grad_norm:
      self.grad_norm.reset()

  @abstractmethod
  def compute_loss(self, losses, external_weights, split):
    raise NotImplementedError

class UWLoss(MTLLoss):
  def __init__(self, config, tasks):
    super().__init__(config, tasks)

    self.sigma = nn.Parameter(torch.zeros(self.losses_dim))

  # cannot distinguish between internal and external weights
  # todo modulate gradient?
  def compute_loss(self, losses, external_weights, split):
    weights = self.sigma.exp()
    return 0.5 * (losses / weights ** 2).sum() + weights.prod().log()

class DWALoss(MTLLoss):
  def __init__(self, config, tasks):
    super().__init__(config, tasks)

    self.history = []

  def compute_loss(self, losses, external_weights, split):
    if len(self.history) == 2:
      ratios = self.history[1] / (self.history[0] + 1e-8)
    else:
      ratios = torch.ones(self.losses_dim).cuda()

    if split == "train" and self.config["mtl_weighing"] == "dwa":
      self.history.append(losses.detach())
      if len(self.history) > 2:
        self.history.pop(0)

    weights = F.softmax(ratios / self.config["mtl_dwa_T"], dim=0)
    return losses @ (weights * external_weights)
    
class RWLoss(MTLLoss):
  def __init__(self, config, tasks):
    super().__init__(config, tasks)

  def compute_loss(self, losses, external_weights, split):
    weights = F.softmax(torch.rand(self.losses_dim, device="cuda"), dim=0)
    return losses @ (weights * external_weights)
  
_constructors = {
  "uw": UWLoss,
  "dwa": DWALoss,
  "rw": RWLoss,
}

def construct_mtl_loss(config, tasks):
  if config["mtl_weighing"] not in _constructors:
    raise ValueError(f"invalid option {config['mtl_weighing']=}")

  constructor = _constructors[config["mtl_weighing"]]
  return constructor(config, tasks)
