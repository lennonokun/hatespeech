from typing import *
from hydra_zen import builds
from pydantic import BaseModel

import numpy as np
import pandas as pd
import json
# from pathlib import Path

from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, WeightedRandomSampler, RandomSampler

from .tasks import TaskSet

class DatasetInfo(BaseModel):
  dataset_path: str
  cats: Dict[str, List[str]]

  @property
  def cols(self):
    return {k: [f"{k}_{x}" for x in v] for k,v in self.cats.items()}

  @classmethod
  def from_config(cls, col_info_path: str, dataset_path: str):
    return cls(dataset_path=dataset_path, cats=json.load(open(col_info_path, "r")))

# TODO shared feature
# batch sampled
class CombinedDataset(Dataset):
  def __init__(self, datasets):
    self.names = list(datasets.keys())
    self.datasets = [datasets[name] for name in self.names]
    self.num_datasets = len(self.datasets)

    lengths = np.array([len(dataset) for dataset in self.datasets])
    self.length = np.sum(lengths)
    self.offsets = np.cumsum(lengths) - lengths
    self.weights = np.concatenate([np.full(length, 1/length) for length in lengths])
    self.min_length = int(np.min(lengths))

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    # get indices per name
    diffs = np.array(idx)[:, None] - self.offsets[None, :]
    name_reverse_idx = np.argmax(diffs[:, ::-1] >= 0, axis=1)
    name_idx = (self.num_datasets - 1) - name_reverse_idx

    # group batches 
    batches = {}
    for i in range(self.num_datasets):
      curr_name_idx = np.where(name_idx == i)[0]
      dataset_idx = diffs[curr_name_idx, i]
      batches[self.names[i]] = self.datasets[i][dataset_idx]
      batches[self.names[i]]["size"] = len(dataset_idx)
    return batches
  
# batch sampled
class HateDataset(Dataset):
  name: ClassVar[str]
  multis: ClassVar[List[str]]
  unis: ClassVar[List[str]]
  
  def __init__(self,
    batch_size: int,
    dynamic_length: bool,
    max_length: int,
    info: DatasetInfo,
    split: str,
  ):
    self.batch_size = batch_size
    self.dynamic_length = dynamic_length
    self.max_length = max_length
    self.df = pd.read_parquet(info.dataset_path.format(name=self.name, split=split))

    self.locs = {}
    for var in self.multis:
      self.locs[var] = [self.df.columns.get_loc(col) for col in info.cols[var]]
    for var in self.unis:
      self.locs[var] = self.df.columns.get_loc(var)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    raise NotImplementedError
    
class ExplainDataset(HateDataset):
  name = "explain"
  multis = ["label", "target"]
  unis = ["tokens", "mask", "rationale"]

  def __getitem__(self, idx):
    num_idx = len(idx)
    if num_idx == 0:
      return {}
    
    label = np.array(self.df.iloc[idx, self.locs["label"]], dtype=np.float32)
    target = np.array(self.df.iloc[idx, self.locs["target"]], dtype=np.float32)

    if self.dynamic_length:
      tokens_list = self.df.iloc[idx, self.locs["tokens"]]
      max_len = max(len(tokens) for tokens in tokens_list)
    else:
      max_len = self.max_length

    tokens = np.zeros((num_idx, max_len), dtype=np.int32)
    mask = np.zeros((num_idx, max_len), dtype=np.bool_)
    rationale = np.zeros((num_idx, max_len), dtype=np.float32)
    for batch, i in enumerate(idx):
      length = len(self.df.iat[i, self.locs["tokens"]])
      tokens[batch, :length] = self.df.iat[i, self.locs["tokens"]]
      for pos, val in self.df.iat[i, self.locs["rationale"]]:
        rationale[batch, pos] = val
      for start, end in self.df.iat[i, self.locs["mask"]]:
        mask[batch, start: end] = True

    return {
      "tokens": tokens,
      "mask": mask,
      "label": label,
      "target": target,
      "rationale": rationale,
    }

class MeasuringDataset(HateDataset):
  name = "measuring"
  multis = []
  unis = ["score", "tokens", "mask"]
    
  def __getitem__(self, idx):
    num_idx = len(idx)
    if num_idx == 0:
      return {}
    
    score = np.array(self.df.iloc[idx, self.locs["score"]], dtype=np.float32)

    if self.dynamic_length:
      tokens_list = self.df.iloc[idx, self.locs["tokens"]]
      max_len = max(len(tokens) for tokens in tokens_list)
    else:
      max_len = self.max_length

    tokens = np.zeros((num_idx, max_len), dtype=np.int32)
    mask = np.zeros((num_idx, max_len), dtype=np.bool_)
    for batch, i in enumerate(idx):
      length = len(self.df.iat[i, self.locs["tokens"]])
      tokens[batch, :length] = self.df.iat[i, self.locs["tokens"]]
      for start, end in self.df.iat[i, self.locs["mask"]]:
        mask[batch, start: end] = True

    return {
      "score": score,
      "tokens": tokens,
      "mask": mask,
    }

class HateDatamodule(LightningDataModule):
  _constructors_list: List[type[HateDataset]] = [ExplainDataset, MeasuringDataset]
  _constructors_dict = {x.name: x for x in _constructors_list}

  def __init__(
    self,
    batch_size: int,
    dynamic_length: bool,
    max_length: int,
    info: DatasetInfo,
    tasks: TaskSet
  ):
    super().__init__()
    self.batch_size = batch_size
    self.dynamic_length = dynamic_length
    self.max_length = max_length
    self.info = info
    self.tasks = tasks
  
  def _select_data(self, split):
    return CombinedDataset({
      name: self._constructors_dict[name](
        batch_size=self.batch_size,
        dynamic_length=self.dynamic_length,
        max_length=self.max_length,
        info=self.info,
        split=split,
      )
      for name in self.tasks.datasets()
    })
  
  def setup(self, stage: str):
    self.datasets = {}
    for split in ["train", "valid", "test"]:
      self.datasets[split] = self._select_data(split)

  def _get_dataloader(self, split: str):
    if split == "train":
      random_sampler = WeightedRandomSampler(
        weights=self.datasets[split].weights,
        num_samples=self.datasets[split].min_length,
      )
    else:
      random_sampler = RandomSampler(self.datasets[split])
    batch_sampler = BatchSampler(
      random_sampler,
      batch_size=self.batch_size,
      drop_last=True
    )
    return DataLoader(
      self.datasets[split],
      sampler=batch_sampler,
      batch_size=None,
      num_workers=4,
      pin_memory=True,
    )

  def train_dataloader(self):
    return self._get_dataloader("train")
  
  def val_dataloader(self):
    return self._get_dataloader("valid")
  
  def test_dataloader(self):
    return self._get_dataloader("test")

DatasetInfoCfg = builds(
  DatasetInfo.from_config,
  col_info_path="data/col_info.json",
  dataset_path="data/{name}/output_{split}.parquet"
)
HateDatamoduleCfg = builds(
  HateDatamodule,
  batch_size = 64,
  max_length = 128,
  dynamic_length = True,
  info=DatasetInfoCfg,
  tasks="${tasks}",
)

# class PartialDatamodule(Protocol):
#   def __call__(self, tasks: TaskSet) -> HateDatamodule: ...
