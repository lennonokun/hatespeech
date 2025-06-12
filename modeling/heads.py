from typing import *
from pydantic import BaseModel
from abc import ABC, abstractmethod

import json
import torch as pt
from torch import nn
from transformers import PreTrainedModel
from torchmetrics import MetricCollection
from torchmetrics import classification as clf, regression as reg
from hydra_zen import store

from .utils import *
from .custom import MaskedBinaryF1, MyBCELoss
from .tasks import TaskSet

class Stats(BaseModel):
  label_freqs: List[float]
  target_freqs: List[float]
  rationale_freq: float

  @classmethod
  def from_json(cls, path: str):
    return cls(**json.load(open(path, "r")))

class HateHead(ABC, nn.Module):
  # implemented by subclass
  name: ClassVar[str]
  loss_args: ClassVar[List[str]]
  metrics_args: ClassVar[List[str]]
  
  def __init__(
    self,
    dropout: float,
    shape: List[int],
    hidden_size: int,
    output_dim: int,
    stats: Stats,
  ):
    ABC.__init__(self)
    nn.Module.__init__(self)

    self.dropout = dropout
    self.shape = shape
    self.hidden_size = hidden_size
    self.output_dim = output_dim
    self.stats = stats

    self.loss = self.make_loss()
    metrics = self.make_metrics()
    self.metrics = {
      split: metrics.clone(prefix=f"{split}_").cuda()
      for split in ["train", "valid", "test"]
    }
    self.head = self.make_head()

  def make_head(self) -> nn.Module:
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

    args += [
      nn.Dropout(self.dropout),
      nn.Linear(prev_dim, self.output_dim)
    ]
    return nn.Sequential(*args)

  @abstractmethod
  def make_loss(self) -> nn.Module:
    raise NotImplementedError

  @abstractmethod
  def make_metrics(self) -> MetricCollection:
    raise NotImplementedError

  @abstractmethod
  def forward(self, hidden, batch):
    raise NotImplementedError

  def compute_metrics(self, results, split):
    return self.metrics[split](*[results[arg] for arg in self.metrics_args])

  def compute_loss(self, results):
    return self.loss(*[results[arg] for arg in self.loss_args])

  def compute(self, hidden, batch, split):
    results = self.forward(hidden, batch)
    metrics = self.compute_metrics(results, split)
    loss = self.compute_loss(results)
    return metrics, loss

class RationaleHead(HateHead):
  name = "rationale"
  loss_args = ["logits", "trues", "masks"]
  metrics_args = ["preds", "hards", "masks"]
  
  def make_loss(self):
    return MyBCELoss(freq=self.stats.rationale_freq)

  def make_metrics(self):
    return MetricCollection({"rationale_f1": MaskedBinaryF1()})
    
  def forward(self, hidden, batch):
    logits = self.head(hidden).squeeze(-1)
    return {
      "logits": logits,
      "trues": batch["rationale"],
      "preds": pt.gt(logits, 0),
      "hards": pt.gt(batch["rationale"], 0.5),
      "masks": batch["mask"] & (batch["label"][:, 1].gt(0)[:, None]),
    }

class TargetHead(HateHead):
  name = "target"
  loss_args = ["logits", "trues"]
  metrics_args = ["preds", "hards"]
  
  def make_loss(self):
    return MyBCELoss(freq=self.stats.target_freqs)

  def make_metrics(self):
    return MetricCollection({"target_f1": clf.MultilabelF1Score(
      num_labels=self.output_dim, average="micro"
    )})

  def forward(self, hidden, batch):
    logits = self.head(hidden[:, 0, :])
    return {
      "logits": logits,
      "trues": batch["target"],
      "preds": pt.gt(logits, 0),
      "hards": pt.gt(batch["target"], 0.5),
    }

class LabelHead(HateHead):
  name = "label"
  loss_args = ["logits", "trues"]
  metrics_args = ["preds", "hards"]
  
  def make_loss(self):
    return MyBCELoss(freq=self.stats.label_freqs)

  def make_metrics(self):
    return MetricCollection({"label_f1": clf.MulticlassF1Score(
      num_classes=self.output_dim, average="macro"
    )})

  def forward(self, hidden, batch):
    logits = self.head(hidden[:, 0, :])
    return {
      "logits": logits,
      "trues": batch["label"],
      "preds": pt.argmax(logits, dim=-1),
      "hards": pt.argmax(batch["label"], dim=-1)
    }

class ScoreHead(HateHead):
  name = "score"
  loss_args = ["preds", "trues"]
  metrics_args = ["preds", "trues"]

  def make_loss(self):
    return nn.MSELoss()

  def make_metrics(self):
    return MetricCollection({"score_mse": reg.MeanSquaredError()})
  
  def forward(self, hidden, batch):
    return {
      "preds": self.head(hidden[:, 0, :]).squeeze(-1),
      "trues": batch["score"],
    }

class HateHeads(nn.Module):
  _constructors_list = [TargetHead, RationaleHead, LabelHead, ScoreHead]
  _constructors_dict = {x.name: x for x in _constructors_list}

  # TODO TAKE ENCODER
  def __init__(
    self,
    dropout: float,
    shape: List[int],
    tasks: TaskSet,
    model: PreTrainedModel,
    stats: Stats
  ):
    super().__init__()
    mapping = {}
    for name, task in tasks.items():
      constructor = self._constructors_dict.get(name)
      if task is None or constructor is None:
        raise ValueError(f"invalid task name: {name}")

      mapping[name] = constructor(
        dropout = dropout,
        shape = shape,
        hidden_size = model.config.hidden_size,
        output_dim = task.output_dim,
        stats = stats,
      )
    self.mapping = nn.ModuleDict(mapping)

  def __getitem__(self, name: str) -> HateHead:
    return cast(HateHead, self.mapping[name])

StatsCfg = fbuilds(Stats.from_json, path="data/stats.json")

heads_store = store(group="heads")
heads_store(fbuilds(
  HateHeads,
  dropout=0.2,
  shape=[128, 128],
  model="${model}",
  stats=StatsCfg, 
  tasks="${tasks}",
), name="default")
