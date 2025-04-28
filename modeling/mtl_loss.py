from abc import ABC, abstractmethod

import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

class MTLGradNorm(nn.Module):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.coefs = None
    self.history = []
    self.wait = 0

  def reset(self):
    self.coefs = None
    self.history = []

  def compute_norms(self, losses_flat, norm_layers):
    norms = []
    for loss in losses_flat:
      grad = torch.autograd.grad(
        loss, norm_layers, create_graph=True, retain_graph=True
      )[0].detach()
      norms.append(torch.norm(grad))
    return torch.stack(norms)

  def forward(self, losses_flat, norm_layers, split):
    losses_norm = losses_flat
    if self.coefs is not None:
      losses_norm *= self.coefs 
    
    if split == "train":
      self.wait += 1

      if self.wait == self.config["mtl_norm_period"]:
        self.history.append(self.compute_norms(losses_flat, norm_layers))

        if len(self.history) > self.config["mtl_norm_length"]:
          self.history.pop(0)
        
        mean_norms = torch.mean(torch.stack(self.history), dim=0) 
        self.coefs = torch.mean(mean_norms) / (mean_norms + 1e-8)
        self.wait = 0

    return losses_norm

class MTLLoss(ABC, nn.Module):
  def __init__(self, config, tasks):
    ABC.__init__(self)
    nn.Module.__init__(self)

    self.config = config
    self.losses_dim = sum(task.loss_dim for task in tasks.values())
    self.loss_importances = torch.Tensor(np.concatenate([
      np.full(task.loss_dim, task.importance) for task in tasks.values()
    ], axis=0)).cuda()
    self.loss_importances *= self.losses_dim / torch.sum(self.loss_importances)
    self.grad_norm = MTLGradNorm(config) if config["mtl_norm_do"] else None

  def forward(self, losses, norm_layers, split):
    losses_flat = torch.cat([torch.atleast_1d(loss) for loss in losses], dim=0)
    losses_norm = self.grad_norm(losses_flat, norm_layers, split) \
      if self.grad_norm else losses_flat

    return self.compute_loss(losses_flat, losses_norm, split)

  def reset_norms(self):
    if self.grad_norm:
      self.grad_norm.reset()

  @abstractmethod
  def compute_loss(self, losses_flat, losses_norm, split):
    raise NotImplementedError

class UWLoss(MTLLoss):
  def __init__(self, config, tasks):
    super().__init__(config, tasks)

    self.sigma = nn.Parameter(torch.zeros(self.losses_dim))

  # distinction between norm and flat losss meaningless, bc it is differentiable
  def compute_loss(self, losses_flat, losses_norm, split):
    weights = self.sigma.exp()
    return 0.5 * (losses_flat / weights ** 2) + weights.prod().log()

class DWALoss(MTLLoss):
  def __init__(self, config, tasks):
    super().__init__(config, tasks)

    self.history = []

  def compute_loss(self, losses_flat, losses_norm, split):
    if len(self.history) == 2:
      ratios = self.history[1] / (self.history[0] + 1e-8)
    else:
      ratios = torch.ones(self.losses_dim).cuda()

    if split == "train" and self.config["mtl_weighing"] == "dwa":
      self.history.append(losses_flat.detach())
      if len(self.history) > 2:
        self.history.pop(0)

    weights = F.softmax(ratios / self.config["mtl_dwa_T"], dim=0)
    return losses_norm @ (weights * self.loss_importances)
    
class RWLoss(MTLLoss):
  def __init__(self, config, tasks):
    super().__init__(config, tasks)

  def compute_loss(self, losses_flat, losses_norm, split):
    weights = F.softmax(torch.rand(self.losses_dim, device="cuda"), dim=0)
    return losses_norm @ (weights * self.loss_importances)
  
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

# class MTLLoss(nn.Module):
#   def __init__(self, config, tasks):
#     super().__init__()
#     self.config = config
#     self.losses_dim = sum(task.loss_dim for task in tasks.values())
#     self.loss_importances = torch.Tensor(np.concatenate([
#       np.full(task.loss_dim, task.importance) for task in tasks.values()
#     ], axis=0)).cuda()
#     self.loss_importances *= self.losses_dim / torch.sum(self.loss_importances)

#     self.loss_norm = None 
#     self.norm_history = []
#     self.history_wait = 0

#     self.dwa_loss_history = []

#   def reset_norms(self):
#     self.loss_norm = None
#     self.norm_history = []
  
#   def compute_norms(self, losses_flat, norm_layers):
#     norms = []
#     for loss in losses_flat:
#       grad = torch.autograd.grad(
#         loss, norm_layers, create_graph=True, retain_graph=True
#       )[0].detach()
#       norms.append(torch.norm(grad))
#     return torch.stack(norms)
  
#   def forward(self, losses, norm_layers, split):
#     losses_flat = torch.cat([torch.atleast_1d(loss) for loss in losses], dim=0)
#     losses_norm = losses_flat * (1 if self.loss_norm is None else self.loss_norm)
          
#     if split == "train" and self.config["mtl_norm_do"]:
#       self.history_wait += 1

#       if self.history_wait == self.config["mtl_norm_period"]:
#         self.norm_history.append(self.compute_norms(losses_flat, norm_layers))

#         if len(self.norm_history) > self.config["mtl_norm_length"]:
#           self.norm_history.pop(0)
        
#         mean_norms = torch.mean(torch.stack(self.norm_history), dim=0) 
#         self.loss_norm = torch.mean(mean_norms) / (mean_norms + 1e-8)
#         self.history_wait = 0
    
#     # DWA
#     if len(self.dwa_loss_history) == 2:
#       ratios = self.dwa_loss_history[1] / (self.dwa_loss_history[0] + 1e-8)
#     else:
#       ratios = torch.ones(self.losses_dim).cuda()

#     if split == "train" and self.config["mtl_weighing"] == "dwa":
#       self.dwa_loss_history.append(losses_flat.detach())
#       if len(self.dwa_loss_history) > 2:
#         self.dwa_loss_history.pop(0)

#     softmax = F.softmax(ratios / self.config["mtl_dwa_T"], dim=0)
#     return losses_norm @ (softmax * self.loss_importances)

#     # weights = F.softmax(torch.rand(self.losses_dim, device="cuda"), dim=0)
#     # return losses_norm @ (weights * self.loss_importances)
