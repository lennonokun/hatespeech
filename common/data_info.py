from typing import *
from pydantic import BaseModel, computed_field
import json
from collections import ChainMap

from .config import fbuilds

class DataStats(BaseModel):
  label_freqs: List[float]
  target_freqs: List[float]
  rationale_freq: float

  @classmethod
  def from_jsons(cls, paths: List[str]):
    loadeds = [json.load(open(path, "r")) for path in paths]
    return cls(**ChainMap(*loadeds))

class DataInfo(BaseModel):
  all_dataset_names: List[str]
  explain_dirty_path: str
  input_dataset_path: str
  output_dataset_path: str
  output_stats_path: str
  cats: Dict[str, List[str]]

  @computed_field
  @property
  def stats(self) -> DataStats:
    paths = [self.output_stats_path.format(name=name) for name in self.all_dataset_names]
    return DataStats.from_jsons(paths)

  @computed_field
  @property
  def cols(self) -> Dict[str, List[str]]:
    return {k: [f"{k}_{x}" for x in v] for k,v in self.cats.items()}

DataInfoCfg = fbuilds(
  DataInfo,
  all_dataset_names=["explain", "measuring"],
  explain_dirty_path="data/explain/dirty.json",
  input_dataset_path="data/{name}/input.parquet",
  output_dataset_path="data/{name}/output_{split}.parquet",
  output_stats_path="data/{name}/stats.json",
  cats={
    "target": [
      "African", "Arab", "Asian", "Caucasian", "Hispanic",
      "Homosexual", "Islam", "Jewish", "Other", "Refugee", "Women"
    ], "label": [
      "hatespeech", "offensive", "normal",
    ]
  }
)
