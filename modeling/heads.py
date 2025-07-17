from typing import *
from pydantic import BaseModel
from abc import ABC, abstractmethod
from functools import partial

import json
import numpy as np
import torch as pt
from torch import nn
from transformers import PreTrainedModel
from torchmetrics import classification as clf, regression as reg
from hydra_zen import store

from .utils import *
from .custom import MaskedBinaryF1, MyBCELoss, FocalLoss, ExtendedMetric, ExtendedMetricSet
from .tasks import TaskSet

class Stats(BaseModel):
  label_freqs: List[float]
  target_freqs: List[float]
  rationale_freq: float

  @classmethod
  def from_json(cls, path: str):
    return cls(**json.load(open(path, "r")))

class HateHead(ABC, nn.Module):
  name: ClassVar[str]
  loss_args: ClassVar[List[str]]

  def __init__(
    self,
    dropout: float,
    shape: List[int],
    hidden_size: int,
    output_dim: int,
    mask,
    shrink_output: bool,
    stats: Stats,
  ):
    self.stats = stats
    ABC.__init__(self)
    nn.Module.__init__(self)

    self.dropout = dropout
    self.shape = shape
    self.hidden_size = hidden_size
    self.output_dim = output_dim
    self.mask = mask
    self.shrink_output = shrink_output

    self.model = self.make_model()
    self.loss = self.make_loss()
    metrics = self.make_metrics()
    self.metrics = {
      split: metrics.clone(prefix=f"{split}_").cuda()
      for split in ["train", "valid", "test"]
    }

  def make_model(self) -> nn.Module:
    args = []
    prev_dim = self.hidden_size

    for dim in self.shape:
      args += [
        nn.Dropout(self.dropout),
        nn.Linear(prev_dim, dim),
        nn.ReLU(),
        nn.LayerNorm([dim]), # TODO input_dims
      ]
      prev_dim = dim

    output_dim = int(np.sum(self.mask)) if self.shrink_output else self.output_dim
    args += [
      nn.Dropout(self.dropout),
      nn.Linear(prev_dim, output_dim),
    ]
    return nn.Sequential(*args)

  def apply_mask_to_labels(self, labels):
    return [label for mask_bit, label in zip(self.mask, labels) if mask_bit]

  @abstractmethod
  def make_loss(self) -> nn.Module:
    raise NotImplementedError

  @abstractmethod
  def make_metrics(self) -> ExtendedMetricSet:
    raise NotImplementedError

  @abstractmethod
  def forward(self, output, batch):
    raise NotImplementedError

  def compute_loss(self, results):
    return self.loss(*[results[arg] for arg in self.loss_args])

  def compute(self, output, batch, split):
    results = self.forward(output, batch)
    loss = self.compute_loss(results)
    log_metrics = partial(self.metrics[split].log_all, results=results)
    return log_metrics, loss

class RationaleHead(HateHead):
  name = "rationale"
  loss_args = ["logits", "trues", "masks"]
  
  def make_loss(self):
    return FocalLoss(alpha=0.8, gamma=2.0)

  def make_metrics(self):
    return ExtendedMetricSet(data={
      "rationale_f1": ExtendedMetric(
        metric=MaskedBinaryF1(),
        args=["preds", "hards", "masks"],
      )
    })
    
  def forward(self, output, batch):
    hidden = output.last_hidden_state.float()
    logits = self.model(hidden).squeeze(-1)
    label_mask = batch["label"][:, [0,1]].gt(0).any(dim=1)[:, None]
    return {
      "logits": logits,
      "trues": batch["rationale"],
      "preds": pt.gt(logits, 0),
      "hards": pt.gt(batch["rationale"], 0.5),
      "masks": batch["mask"] & label_mask,
    }

# MASKABLE
class TargetHead(HateHead):
  name = "target"
  loss_args = ["logits", "trues"]
  labels = [
    "african", "arab", "asian", "caucasian", "hispanic",
    "homosexual", "islam", "jewish", "other", "refugee", "women"
  ]
  
  def make_loss(self):
    freq = np.array(self.stats.target_freqs)[self.mask]
    return MyBCELoss(freq=freq, reduce_dim=0)

  def make_metrics(self):
    return ExtendedMetricSet(data={
      "target_macro_f1": ExtendedMetric(
        metric=clf.MultilabelF1Score(
          num_labels=int(np.sum(self.mask)), average="macro"
        ), args=["preds", "hards"],
      ), "target_micro_f1": ExtendedMetric(
        metric=clf.MultilabelF1Score(
          num_labels=int(np.sum(self.mask)), average="micro"
        ), args=["preds", "hards"],
      ), "target_f1_{}": ExtendedMetric(
        metric=clf.MultilabelF1Score(
          num_labels=int(np.sum(self.mask)), average="none"
        ), args=["preds", "hards"],
        labelss=[self.apply_mask_to_labels(self.labels)]
      ), "target_confusion_{}_{}_{}": ExtendedMetric(
        metric=clf.MultilabelConfusionMatrix(
          num_labels=int(np.sum(self.mask)),
        ), args=["preds", "hards"],
        labelss=[
          self.apply_mask_to_labels(self.labels),
          ["neg", "pos"],
          ["neg", "pos"],
        ], reduce_fx="sum",
      )
    })

  def forward(self, output, batch):
    hidden = output.last_hidden_state.float()
    logits = self.model(hidden[:, 0, :])
    trues = batch["target"][:, self.mask]
    return {
      "logits": logits,
      "trues": trues,
      "preds": pt.gt(logits, 0),
      "hards": pt.gt(trues, 0.5),
    }

class LabelHead(HateHead):
  name = "label"
  loss_args = ["logits", "trues"]
  
  def make_loss(self):
    return MyBCELoss(freq=self.stats.label_freqs)

  def make_metrics(self):
    return ExtendedMetricSet(data={
      "label_f1": ExtendedMetric(
        metric=clf.MulticlassF1Score(
          num_classes=self.output_dim, average="macro"
        ), args=["preds", "hards"]
      )
    })

  def forward(self, output, batch):
    hidden = output.last_hidden_state.float()
    logits = self.model(hidden[:, 0, :])
    return {
      "logits": logits,
      "trues": batch["label"],
      "preds": pt.argmax(logits, dim=-1),
      "hards": pt.argmax(batch["label"], dim=-1)
    }

class ScoreHead(HateHead):
  name = "score"
  loss_args = ["preds", "trues"]

  def make_loss(self):
    return nn.MSELoss()

  def make_metrics(self):
    return ExtendedMetricSet(data={
      "score_mse": ExtendedMetric(
        metric=reg.MeanSquaredError(),
        args=["preds", "trues"],
      )
    })
  
  def forward(self, output, batch):
    hidden = output.last_hidden_state.float()
    return {
      "preds": self.model(hidden[:, 0, :]).squeeze(-1),
      "trues": batch["score"],
    }

class HateHeads(nn.Module):
  _constructors_list = [TargetHead, RationaleHead, LabelHead, ScoreHead]
  _constructors_dict = {x.name: x for x in _constructors_list}

  def __init__(
    self,
    dropout: float,
    shape: List[int],
    tasks: TaskSet,
    model: PreTrainedModel,
    stats: Stats,
    path: str | None,
    load: bool,
  ):
    super().__init__()
    self.mapping = nn.ModuleDict()
    for name, task in tasks.items():
      constructor = self._constructors_dict.get(name)
      if task is None or constructor is None:
        raise ValueError(f"invalid task name: {name}")

      self.mapping[name] = constructor(
        dropout=dropout,
        shape=shape,
        hidden_size=model.config.hidden_size,
        output_dim=task.output_dim,
        mask=task.mask,
        shrink_output=task.shrink_output,
        stats=stats,
      )

    if load and path is not None:
      self.mapping.load_state_dict(pt.load(path))

  def __getitem__(self, name: str) -> HateHead:
    return cast(HateHead, self.mapping[name])

  def list(self):
    return self.mapping.values()

  def save(self, path: str):
    pt.save(self.mapping.state_dict(), path)

  def load(self, path: str):
    self.load_state_dict(pt.load(path, weights_only=True))

StatsCfg = fbuilds(Stats.from_json, path="data/stats.json")

heads_store = store(group="heads")
shapes = {
  "xsmall": [64, 64],
  "small": [64, 64, 64],
  "medium": [128, 128],
  "large": [128, 128, 128],
  "xlarge": [256, 128, 128],
}
for name, shape in shapes.items():
  heads_store(fbuilds(
    HateHeads,
    dropout=0.2,
    shape=shape,
    model="${model}",
    stats=StatsCfg, 
    tasks="${tasks}",
    path="${load_path}/heads.pt",
    load=False,
  ), name=name)
