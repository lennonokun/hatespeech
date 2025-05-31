from typing import * # pyright: ignore[reportWildcardImportFromLibrary]
from pydantic import BaseModel
# from collections import ChainMap

from hydra_zen import make_custom_builds_fn

class Stats(BaseModel):
  label_freqs: List[float]
  target_freqs: List[float]
  rationale_freq: float

class Task(BaseModel):
  dataset: str
  # monitors: Dict[str, float]
  importance: float
  output_dim: int
  loss_dim: int
  
class TaskSet(BaseModel):
  active: List[str]
  all: Dict[str, Task]

  @property
  def datasets(self) -> List[str]:
    return list(set(task.dataset for task in self.all.values()))

  # @property
  # def monitors(self) -> Dict[str, float]:
  #   nested = (task.monitors for task in self.all.values())
  #   return dict(ChainMap(*nested))

  def get(self, name: str) -> Task:
    if name not in self.all:
      raise ValueError(f"invalid task name: {name}")
    return self.all[name]

  def iter_pairs(self) -> Iterable[Tuple[str, Task]]:
    for name in self.active:
      yield name, self.get(name)

  def iter_tasks(self) -> Iterable[Task]:
    for name in self.active:
      yield self.get(name)

builds = make_custom_builds_fn(populate_full_signature=True)

label_task = Task(
  dataset = "explain",
  # monitors = {"valid_label_f1": 5e-3},
  importance = 1e0,
  output_dim = 3,
  loss_dim = 1
)
target_task = Task(
  dataset = "explain",
  # monitors = {"valid_target_f1": 5e-3},
  importance = 3e0,
  output_dim = 13,
  loss_dim = 13
)
rationale_task = Task(
  dataset = "explain",
  # monitors = {"valid_rationale_f1": 1e-2},
  importance = 1e0,
  output_dim = 1,
  loss_dim = 1,
)
score_task = Task(
  dataset = "measuring",
  # monitors = {"valid_score_mse": -2e-2},
  importance = 1e0,
  output_dim = 1,
  loss_dim = 1,
)
TaskSetCfg = builds(
  TaskSet,
  active=[],
  all={
    "label": label_task,
    "rationale": rationale_task,
    "target": target_task,
    "score": score_task,
  }
)

# TODO read from json?
StatsCfg = builds(
  Stats,
  label_freqs = [0.09819006022103104, 0.09066243134140693, 0.12927668585798424],
  target_freqs = [0.05469525511216994, 0.013053404804447092, 0.006932036264972536, 0.00860300443385613, 0.006104824300178678, 0.03207927999470584, 0.03732380385149891, 0.033535173052743034, 0.013351201111772881, 0.01581629276685858, 0.02658659254847462],
  rationale_freq = 0.03018080459327523
)

