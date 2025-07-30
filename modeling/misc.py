from typing import *
from copy import deepcopy

import numpy as np
import torch as pt
from torch import nn
from torch.nn import functional as F

from torchmetrics import Metric
from lightning import LightningModule

# metric with arguments and matching labels for non-scalar values
# TODO optional one-time check?
# TODO inherit instead of compose?
class ExtendedMetric:
  def __init__(
    self,
    metric: Metric,
    args: List[str],
    labelss: List[List[str]] | None = None,
    reduce_fx: str = "mean"
  ):
    self.metric = metric
    self.args = args
    self.labelss = labelss if labelss is not None else []
    self.reduce_fx = reduce_fx

  def _iter_values(self, name, values):
    if values.ndim != len(self.labelss):
      raise ValueError(
        f"dimension size mismatch: {values.ndim=} does not match {len(self.labelss)=}"
      )
    for i, (dim, labels) in enumerate(zip(values.shape, self.labelss)):
      if dim != len(labels):
        raise ValueError(
          f"dimension mismatch: {dim=} does {len(labels)=} not match at dimension {i}"
        )

    it = np.nditer(values.cpu().numpy(), flags=["multi_index"])
    for value in it:
      active_labels = [labels[i] for i, labels in zip(it.multi_index, self.labelss)]
      yield name.format(*active_labels), value.item() # pyright: ignore
    
  def log(
    self,
    name: str,
    module: LightningModule,
    results: Dict[str, pt.Tensor],
    **kwargs
  ):
    values = self.metric(*[results[arg] for arg in self.args])
    for formatted, value in self._iter_values(name, values):
      module.log(formatted, value, reduce_fx=self.reduce_fx, **kwargs)

  def compute(self, name):
    computed = self.metric.compute()
    return {formatted: value for formatted, value in self._iter_values(name, computed)}
    
  def clone(self):
    return ExtendedMetric(
      self.metric.clone(),
      deepcopy(self.args),
      deepcopy(self.labelss),
      deepcopy(self.reduce_fx),
    )

  def reset(self):
    self.metric.reset()

  def cuda(self):
    self.metric = self.metric.cuda()
    return self

class ExtendedMetricSet:
  def __init__(self, data: Dict[str, ExtendedMetric]):
    self.data = data

  def log_all(self, module: LightningModule, results: Dict[str, pt.Tensor], **kwargs):
    for name, emetric in self.data.items():
      emetric.log(name, module, results, **kwargs)

  def clone(self, prefix=""):
    return ExtendedMetricSet(data={
      f"{prefix}{name}": emetric.clone()
      for name, emetric in self.data.items()
    })

  def compute(self):
    return {k: v for name, emetric in self.data.items() for k, v in emetric.compute(name).items()}
  
  def cuda(self):
    self.data = {name: emetric.cuda() for name, emetric in self.data.items()}
    return self

  def reset(self):
    for emetric in self.data.values():
      emetric.reset()
    
# sourced from https://github.com/kardasbart/MultiLR
class MultiLR(pt.optim.lr_scheduler._LRScheduler):
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
    self.add_state("correct", default=pt.tensor(0.), dist_reduce_fx="sum")
    self.add_state("incorrect", default=pt.tensor(0.), dist_reduce_fx="sum")

  def update(self, preds: pt.Tensor, labels: pt.Tensor, mask: pt.Tensor):
    preds, labels, mask = preds.bool(), labels.bool(), mask.bool()
    self.correct += ((preds == labels) & mask).sum().float()
    self.incorrect += ((preds != labels) & mask).sum().float()

  def compute(self):
    return self.correct / (self.correct + self.incorrect + 1e-8)

class MaskedBinaryF1(Metric):
  def __init__(self, dist_sync_on_step=False):
    super().__init__(dist_sync_on_step=dist_sync_on_step)
    self.add_state("true_positives", default=pt.tensor(0.), dist_reduce_fx="sum")
    self.add_state("true_negatives", default=pt.tensor(0.), dist_reduce_fx="sum")
    self.add_state("false_positives", default=pt.tensor(0.), dist_reduce_fx="sum")
    self.add_state("false_negatives", default=pt.tensor(0.), dist_reduce_fx="sum")

  def update(self, preds: pt.Tensor, labels: pt.Tensor, mask: pt.Tensor):
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
    probs = pt.sigmoid(logits)
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
      self.pos_weight = pt.Tensor((1 - freq) / (freq + 1e-8)).cuda()

  def forward(self, logits, labels, mask=None):
    loss = F.binary_cross_entropy_with_logits(
      logits, labels, reduction="none", pos_weight=self.pos_weight,
    )

    return reduce_loss_apply_mask(loss, mask, self.reduce_dim)

