from typing import *
from pydantic import BaseModel

from hydra_zen import store
from .utils import *

class Task(BaseModel):
  dataset: str
  monitor: Tuple[str, float]
  importance: float
  output_dim: int
  loss_dim: int
  
# TODO maybe UserDict?
class TaskSet:
  def __init__(self, **kwargs: Task):
    self.data = kwargs
    if len(self.data) == 0:
      raise ValueError("TaskSet cannot be empty, please specify tasks")
  
  def datasets(self) -> List[str]:
    return list(set(task.dataset for task in self.values()))

  def monitors(self) -> Dict[str, float]:
    return dict(task.monitor for task in self.values())

  def get(self, name: str) -> Task:
    if name not in self.data:
      raise ValueError(f"invalid task name: {name}")
    return self.data[name]

  def items(self) -> Iterable[Tuple[str, Task]]:
    return self.data.items()

  def values(self) -> Iterable[Task]:
    return self.data.values()

  def names(self) -> Iterable[str]:
    return self.data.keys()

  def __str__(self) -> str:
    return f"TaskSet({self.data})"
  __repr__ = __str__

LabelTaskCfg = fbuilds(
  Task,
  dataset="explain",
  monitor=("valid_label_f1", 5e-3),
  importance=1e0,
  output_dim=3,
  loss_dim=1
)
TargetTaskCfg = fbuilds(
  Task,
  dataset="explain",
  monitor=("valid_target_f1", 5e-3),
  importance=3e0,
  output_dim=13,
  loss_dim=13,
)
RationaleTaskCfg = fbuilds(
  Task,
  dataset="explain",
  monitor=("valid_rationale_f1", 1e-2),
  importance=1e0,
  output_dim=1,
  loss_dim=1,
)
ScoreTaskCfg = fbuilds(
  Task,
  dataset="measuring",
  monitor=("valid_score_mse", -2e2),
  importance=1e0,
  output_dim=1,
  loss_dim=1,
)

tasks_store = store(group="tasks", to_config=remove_types)
tasks_store(TaskSet, name="none")
tasks_store(fbuilds(TaskSet, label=LabelTaskCfg), name="label")
tasks_store(fbuilds(TaskSet, target=TargetTaskCfg), name="target")
tasks_store(fbuilds(TaskSet, rationale=RationaleTaskCfg), name="rationale")
tasks_store(fbuilds(TaskSet, score=ScoreTaskCfg), name="score")

