from typing import * # pyright: ignore[reportWildcardImportFromLibrary]
from abc import ABC, abstractmethod
from hydra_zen import make_custom_builds_fn

import torch
from torch import nn

from torchmetrics import MetricCollection
from torchmetrics import classification as clf, regression as reg

from transformers import PreTrainedModel

from .custom import MaskedBinaryF1, MyBCELoss
from .tasks import TaskSet, Stats

builds = make_custom_builds_fn(populate_full_signature=True)

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
      "preds": torch.gt(logits, 0),
      "hards": torch.gt(batch["rationale"], 0.5),
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
      "preds": torch.gt(logits, 0),
      "hards": torch.gt(batch["target"], 0.5),
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
      "preds": torch.argmax(logits, dim=-1),
      "hards": torch.argmax(batch["label"], dim=-1)
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
    encoder: PreTrainedModel,
    stats: Stats
  ):
    super().__init__()
    mapping = {}
    for name in tasks.active:
      task = tasks.get(name)
      constructor = self._constructors_dict.get(name)
      if task is None or constructor is None:
        raise ValueError(f"invalid task name: {name}")

      mapping[name] = constructor(
        dropout = dropout,
        shape = shape,
        hidden_size = encoder.config.hidden_size,
        output_dim = task.output_dim,
        stats = stats,
      )
    self.mapping = nn.ModuleDict(mapping)

  def __getitem__(self, name: str) -> HateHead:
    return cast(HateHead, self.mapping[name])

HateHeadsCfg = builds(
  HateHeads,
  dropout = 0.2,
  shape = [128, 128],
  zen_partial = True,
)
class PartialHeads(Protocol):
  def __call__(self, tasks: TaskSet, encoder: PreTrainedModel, stats: Stats) -> HateHeads: ...
