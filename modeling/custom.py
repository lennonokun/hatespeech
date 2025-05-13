from dataclasses import dataclass
from typing import *

import numpy as np

import torch
from torch import nn
from torch.nn import functional as F

from lightning.pytorch.callbacks.callback import Callback

from torchmetrics import Metric

# sourced from https://github.com/kardasbart/MultiLR
class MultiLR(torch.optim.lr_scheduler._LRScheduler):
  def __init__(self, optimizer, lambda_factories, last_epoch=-1):
    self.schedulers = []
    values = self._get_optimizer_lr(optimizer)
    for idx, factory in enumerate(lambda_factories):
      self.schedulers.append(factory(optimizer))
      values[idx] = self._get_optimizer_lr(optimizer)[idx]
      self._set_optimizer_lr(optimizer, values)
    super().__init__(optimizer, last_epoch)

  def get_lr(self):
    result = []
    for idx, sched in enumerate(self.schedulers):
      result.append(sched.get_last_lr()[idx])
    return result
  
  @staticmethod
  def _set_optimizer_lr(optimizer, values):
    for param_group, lr in zip(optimizer.param_groups, values):
      param_group['lr'] = lr

  @staticmethod
  def _get_optimizer_lr(optimizer):
    return [group['lr'] for group in optimizer.param_groups]

  def step(self, epoch=None):
    if self.last_epoch != -1:
      values = self._get_optimizer_lr(self.optimizer)
      for idx, sched in enumerate(self.schedulers):
        sched.step()
        values[idx] = self._get_optimizer_lr(self.optimizer)[idx]
        self._set_optimizer_lr(self.optimizer, values)
    super().step()

class MaskedBinaryAccuracy(Metric):
  def __init__(self, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.add_state("correct", default=torch.tensor(0.), dist_reduce_fx="sum")
    self.add_state("incorrect", default=torch.tensor(0.), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    preds, labels, mask = preds.bool(), labels.bool(), mask.bool()
    self.correct += ((preds == labels) & mask).sum().float()
    self.incorrect += ((preds != labels) & mask).sum().float()

  def compute(self):
    return self.correct / (self.correct + self.incorrect + 1e-8)

class MaskedBinaryF1(Metric):
  def __init__(self, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.add_state("true_positives", default=torch.tensor(0.), dist_reduce_fx="sum")
    self.add_state("true_negatives", default=torch.tensor(0.), dist_reduce_fx="sum")
    self.add_state("false_positives", default=torch.tensor(0.), dist_reduce_fx="sum")
    self.add_state("false_negatives", default=torch.tensor(0.), dist_reduce_fx="sum")

  def update(self, preds: torch.Tensor, labels: torch.Tensor, mask: torch.Tensor):
    preds, labels, mask = preds.bool(), labels.bool(), mask.bool()

    self.true_positives += (preds & labels & mask).sum().float()
    self.true_negatives += (~preds & ~labels & mask).sum().float()
    self.false_positives += (preds & ~labels & mask).sum().float()
    self.false_negatives += (~preds & labels & mask).sum().float()

  def compute(self):
    pos_precision = self.true_positives / (self.true_positives + self.false_positives + 1e-8)
    pos_recall = self.true_positives / (self.true_positives + self.false_negatives + 1e-8)
    pos_f1 = 2 * pos_precision * pos_recall / (pos_precision + pos_recall + 1e-8)
    neg_precision = self.true_negatives / (self.true_negatives + self.false_negatives + 1e-8)
    neg_recall = self.true_negatives / (self.true_negatives + self.false_positives + 1e-8)
    neg_f1 = 2 * neg_precision * neg_recall / (neg_precision + neg_recall + 1e-8)
    return (pos_f1 + neg_f1) / 2

def reduce_loss_apply_mask(loss, mask, reduce_dim):
  if mask is not None:
    return (loss * mask).sum(dim=reduce_dim) / (mask.sum(dim=reduce_dim) + 1e-8)
  else:
    return loss.mean(dim=reduce_dim)
  
class FocalLoss(nn.Module):
  def __init__(self, alpha, gamma, reduce_dim=None):
    super().__init__()
    self.alpha = alpha
    self.gamma = gamma
    self.reduce_dim = reduce_dim

  def forward(self, logits, labels, mask=None):
    probs = torch.sigmoid(logits)
    probs_t = probs * labels + (1 - probs) * (1 - labels)
    alpha_t = self.alpha * labels + (1 - self.alpha) * (1 - labels)
    bce_loss = F.binary_cross_entropy_with_logits(logits, labels, reduction="none")
    loss = alpha_t * ((1 - probs_t) ** self.gamma) * bce_loss

    return reduce_loss_apply_mask(loss, mask, self.reduce_dim)

# could normalize multilabel?
class MyBCELoss(nn.Module):
  def __init__(self, freq=None, reduce_dim=None):
    super().__init__()
    self.set_freq(freq)
    self.reduce_dim = reduce_dim

  def set_freq(self, freq):
    if freq is None:
      self.pos_weight = None
    else:
      freq = np.atleast_1d(freq)
      self.pos_weight = torch.Tensor((1 - freq) / (freq + 1e-8)).cuda()

  def forward(self, logits, labels, mask=None):
    loss = F.binary_cross_entropy_with_logits(
      logits, labels, reduction="none", pos_weight=self.pos_weight,
    )

    return reduce_loss_apply_mask(loss, mask, self.reduce_dim)

class MultiEarlyStopping(Callback):
  def __init__(self, monitors: Dict[str, float], patience: float, num_required: int, wait_initial: int):
    super().__init__()
    self.monitors = monitors
    self.patience = patience
    self.num_required = num_required

    self.wait_initial = wait_initial
    self.wait_count = 0
    self.prev_scores = {
      name: float("-inf") * thresh for name, thresh in monitors.items()
    }

  def _check_early_stopping(self, trainer):
    if self.wait_initial > 0:
      self.wait_initial -= 1
      return
    
    num_improved = 0
    for name, thresh in self.monitors.items():
      if name in trainer.callback_metrics:
        curr_score = trainer.callback_metrics[name].squeeze()
        is_improved = ((curr_score - self.prev_scores[name]) / thresh) >= 1.0
        num_improved += bool(is_improved)

    if num_improved >= self.num_required:
      self.wait_count = 0
      for name in self.monitors:
        self.prev_scores[name] = trainer.callback_metrics[name].squeeze()
    else:
      self.wait_count += 1
      if self.wait_count >= self.patience:
        trainer.should_stop = True

  def on_validation_end(self, trainer, pl_module):
    self._check_early_stopping(trainer)
