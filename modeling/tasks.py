from abc import ABC, abstractmethod
from typing import * # pyright: ignore[reportWildcardImportFromLibrary]

import torch
from torch import nn

from torchmetrics import MetricCollection
from torchmetrics import classification as clf
from torchmetrics import regression as reg

from .custom import MaskedBinaryF1, MyBCELoss
  
class BaseTask(ABC, nn.Module):
  # implemented by subclass
  name: ClassVar[str]
  loss_args: ClassVar[List[str]]
  metrics_args: ClassVar[List[str]]
  
  def __init__(
    self,
    config,
    loss_fn,
    metrics,
    loss_dim=1,
    importance=1.0,
    input_dims=None,
  ):
    ABC.__init__(self)
    nn.Module.__init__(self)

    self.loss_fn = loss_fn
    self.metrics = {
      split: metrics.clone(prefix=f"{split}_").cuda()
      for split in ["train", "valid", "test"]
    }
    self.loss_dim = loss_dim
    self.importance = importance

    input_dims = input_dims if input_dims is not None else [] 
    self.head = lambda x: x

  # make task head with configured shape
  def make_head(self, config, input_dims):
    args = []
    prev_dim = config["num_hidden"]
    dims = config["heads"][self.name]

    if len(dims) == 0:
      raise ValueError(f"config['heads']['{self.name}'] must be nonempty.")
    
    # build sequential without last activation + norm
    for dim in config["heads"][self.name]:
      args.append(nn.Linear(prev_dim, dim))
      args.append(nn.ReLU())
      args.append(nn.LayerNorm([dim, *input_dims]))
      prev_dim = dim
    args.pop()
    args.pop()
    return nn.Sequential(*args)

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
  name = "rationale"
  loss_args = ["logits", "trues", "masks"]
  metrics_args = ["preds", "hards", "masks"]
  
  def __init__(self, config):
    super().__init__(
      config = config,
      loss_fn = MyBCELoss(freq=config["stats"]["rationale_freq"]),
      metrics = MetricCollection({"rationale_f1": MaskedBinaryF1()}),
      importance = config["mtl_importances"]["rationale"],
      # TODO make layernorm work 
      # input_dims = [config["batch_size"]],
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
  name = "target"
  loss_args = ["logits", "trues"]
  metrics_args = ["preds", "hards"]
  
  def __init__(self, config):
    if config["mtl_expand_targets"]:
      reduce_dim, loss_dim = 0, config["num_target"]
    else:
      reduce_dim, loss_dim = None, 1

    loss_fn = MyBCELoss(
      freq = config["stats"]["target_freqs"],
      reduce_dim = reduce_dim
    )
    metrics = MetricCollection({"target_f1": clf.MultilabelF1Score(
      num_labels=config["num_target"], average="micro",
    )})

    super().__init__(
      config = config,
      loss_fn = loss_fn,
      metrics = metrics,
      importance = config["mtl_importances"]["target"],
      loss_dim = loss_dim,
    )

  def forward(self, hidden, batch):
    logits = self.head(hidden)
    return {
      "logits": logits,
      "trues": batch["target"],
      "preds": torch.gt(logits, 0),
      "hards": torch.gt(batch["target"], 0.5),
    }

class LabelTask(BaseTask):
  name = "label"
  loss_args = ["logits", "trues"]
  metrics_args = ["preds", "hards"]
  
  def __init__(self, config):
    loss_fn = MyBCELoss(freq=config["stats"]["label_freqs"])
    metrics = MetricCollection({"label_f1": clf.MulticlassF1Score(
      num_classes=config["num_label"], average="macro",
    )})
    
    super().__init__(
      config = config,
      loss_fn = loss_fn,
      metrics = metrics,
      importance = config["mtl_importances"]["label"],
    )

  def forward(self, hidden, batch):
    logits = self.head(hidden)
    return {
      "logits": logits,
      "trues": batch["label"],
      "preds": torch.argmax(logits, dim=-1),
      "hards": torch.argmax(batch["label"], dim=-1)
    }

class ScoreTask(BaseTask):
  name = "score"
  loss_args = ["preds", "trues"]
  metrics_args = ["preds", "trues"]

  def __init__(self, config):
    super().__init__(
      config = config,
      loss_fn = nn.MSELoss(),
      metrics = MetricCollection({"score_mse": reg.MeanSquaredError()}),
      importance = config["mtl_importances"]["score"],
    )

  def forward(self, hidden, batch):
    return {
      "preds": self.head(hidden).squeeze(-1),
      "trues": batch["score"],
    }

_constructors_list: List[BaseTask] = [
  TargetTask,
  RationaleTask,
  LabelTask,
  ScoreTask,
]
_constructors_dict = {
  x.name: x for x in _constructors_list
}

def construct_tasks(config):
  out: Dict[str, BaseTask] = {}
  for name in config["melt_tasks"]:
    constructor = _constructors_dict.get(name)
    if constructor is None:
      raise ValueError(f"invalid element of config['melt_tasks']: {name}")
    else:
      out[name] = constructor(config)

  return out
