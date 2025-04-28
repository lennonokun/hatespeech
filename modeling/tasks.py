from abc import ABC, abstractmethod

import torch
from torch import nn

from torchmetrics import MetricCollection
from torchmetrics import classification as clf
from torchmetrics import regression as reg

from .custom import MaskedBinaryF1, MyBCELoss

class BaseTask(ABC, nn.Module):
  def __init__(
    self,
    loss_fn,
    loss_args,
    metrics,
    metrics_args,
    loss_dim=1,
    importance=1.0
  ):
    ABC.__init__(self)
    nn.Module.__init__(self)

    self.loss_fn = loss_fn
    self.loss_args = loss_args
    self.metrics = {
      split: metrics.clone(prefix=f"{split}_").cuda()
      for split in ["train", "valid", "test"]
    }
    self.metrics_args = metrics_args
    self.loss_dim = loss_dim
    self.importance = importance

  @abstractmethod
  def forward(self, hidden, batch):
    raise NotImplementedError

  def compute_metrics(self, results, split):
    return self.metrics[split](*[results[arg] for arg in self.metrics_args])

  def compute_loss(self, results):
    return self.loss_fn(*[results[arg] for arg in self.loss_args])

  def compute(self, hidden, batch, split):
    results = self.forward(hidden, batch)
    metrics = self.compute_metrics(results, split)
    loss = self.compute_loss(results)
    return metrics, loss

class RationaleTask(BaseTask):
  def __init__(self, config):
    metrics = MetricCollection({"rationale_f1": MaskedBinaryF1()})
    super().__init__(
      loss_fn = MyBCELoss(freq=config["stats"]["rationale_freq"]),
      metrics = metrics,
      loss_args = ["logits", "trues", "masks"],
      metrics_args = ["preds", "hards", "masks"],
      importance = config["mtl_importances"]["rationale"],
    )

    # self.head = nn.Linear(config["num_hidden"], 1)
    self.head = nn.Sequential(
      nn.Linear(config["num_hidden"], 32),
      nn.ReLU(),
      nn.Linear(32, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )

  def forward(self, hidden, batch):
    logits = self.head(hidden).squeeze(-1)
    return {
      "logits": logits,
      "trues": batch["rationale"],
      "preds": torch.gt(logits, 0),
      "hards": torch.gt(batch["rationale"], 0.5),
      "masks": batch["mask"] & (batch["label"][:, 1].gt(0)[:, None]),
    }

class TargetTask(BaseTask):
  def __init__(self, config):
    if config["mtl_expand_targets"]:
      reduce_dim, loss_dim = 0, config["num_target"]
    else:
      reduce_dim, loss_dim = None, 1

    super().__init__(
      loss_fn = MyBCELoss(
        freq = config["stats"]["target_freqs"],
        reduce_dim = reduce_dim
      ),
      metrics = MetricCollection({"target_f1": clf.MultilabelF1Score(
        num_labels=config["num_target"], average="micro",
      )}),
      loss_args = ["logits", "trues"],
      metrics_args = ["preds", "hards"],
      importance = config["mtl_importances"]["target"],
      loss_dim = loss_dim,
    )

    # self.head = nn.Linear(config["num_hidden"], config["num_target"])
    self.head = nn.Sequential(
      nn.Linear(config["num_hidden"], 128),
      nn.ReLU(),
      nn.Linear(128, 64),
      nn.ReLU(),
      nn.Linear(64, config["num_target"])
    )

  def forward(self, hidden, batch):
    logits = self.head(hidden[:, 0, :])
    return {
      "logits": logits,
      "trues": batch["target"],
      "preds": torch.gt(logits, 0),
      "hards": torch.gt(batch["target"], 0.5),
    }

class LabelTask(BaseTask):
  def __init__(self, config):
    super().__init__(
      loss_fn = MyBCELoss(freq=config["stats"]["label_freqs"]),
      metrics = MetricCollection({"label_f1": clf.MulticlassF1Score(
        num_classes=config["num_label"], average="macro",
      )}),
      loss_args = ["logits", "trues"],
      metrics_args = ["preds", "hards"],
      importance = config["mtl_importances"]["label"],
    )

    # self.head = nn.Linear(config["num_hidden"], config["num_label"])
    self.head = nn.Sequential(
      nn.Linear(config["num_hidden"], 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, config["num_label"])
    )

  def forward(self, hidden, batch):
    logits = self.head(hidden[:, 0, :])
    return {
      "logits": logits,
      "trues": batch["label"],
      "preds": torch.argmax(logits, dim=-1),
      "hards": torch.argmax(batch["label"], dim=-1)
    }

class ScoreTask(BaseTask):
  def __init__(self, config):
    super().__init__(
      loss_fn = nn.MSELoss(),
      metrics = MetricCollection({"score_mse": reg.MeanSquaredError()}),
      loss_args = ["preds", "trues"],
      metrics_args = ["preds", "trues"],
      importance = config["mtl_importances"]["score"],
    )

    # self.head = nn.Linear(config["num_hidden"], 1)
    self.head = nn.Sequential(
      nn.Linear(config["num_hidden"], 64),
      nn.ReLU(),
      nn.Linear(64, 32),
      nn.ReLU(),
      nn.Linear(32, 1)
    )

  def forward(self, hidden, batch):
    return {
      "preds": self.head(hidden[:, 0, :]).squeeze(-1),
      "trues": batch["score"],
    }

_constructors = {
  "target": TargetTask,
  "rationale": RationaleTask,
  "label": LabelTask,
  "score": ScoreTask,
}

def construct_tasks(config):
  out = {}
  for name in config["melt_tasks"]:
    if name not in _constructors:
      raise ValueError(f"invalid element of config['melt_tasks']: {name}")
    out[name] = _constructors[name](config)
  return out
