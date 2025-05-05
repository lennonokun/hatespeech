from typing import *

import numpy as np
import pandas as pd
import torch

from lightning import LightningDataModule
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.sampler import BatchSampler, RandomSampler

# TODO shared feature
# batch sampled
class DatasetCombinedDataset(Dataset):
  def __init__(self, config, datasets):
    self.names = list(datasets.keys())
    self.datasets = [datasets[name] for name in self.names]
    self.num_datasets = len(self.datasets)
    # self.features = features

    lengths = np.array([len(dataset) for dataset in self.datasets])
    self.length = np.sum(lengths)
    self.offsets = np.cumsum(lengths) - lengths

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

class TaskCombinedDataset(Dataset):
  def __init__(self, config, datasets):
    self.config = config
    self.datasets = datasets

    lengths = np.array([len(datasets[name]) for name in config["melt_datasets"]])
    self.length = np.sum(lengths)
    self.offsets = np.cumsum(lengths) - lengths

  def __len__(self):
    return self.length

  def __getitem__(self, idx):
    diffs = np.array(idx)[:, None] - self.offsets[None, :]
    # get indices per name
    name_reverse_idx = np.argmax(diffs[:, ::-1] >= 0, axis=1)
    name_idx = (len(self.config["melt_tasks"]) - 1) - name_reverse_idx

    batches = {task: {} for task in self.config["melt_tasks"]}
    for i, (dataset, task) in enumerate(self.config["melt_pairs"]):
      curr_name_idx = np.where(name_idx == i)[0]
      dataset_idx = diffs[curr_name_idx, i]

      if len(dataset_idx) > 0:
        rows = self.datasets[dataset][dataset_idx]
        for feature in self.config["features"]:
          batches[task][feature] = rows[feature]
        batches[task][task] = rows[task]

      batches[task]["size"] = len(dataset_idx)

    return batches
  
# batch sampled
class HateDataset(Dataset):
  def __init__(self, config: dict, name: str, split: str, multis: List[str], unis: List[str]):
    self.config = config
    
    path = self.config["output_dataset_path"].format(name=name, split=split)
    self.df = pd.read_parquet(path)

    self.locs = {}
    for var in multis:
      self.locs[var] = [self.df.columns.get_loc(col) for col in self.config[f"cols_{var}"]]
    for var in unis:
      self.locs[var] = self.df.columns.get_loc(var)

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    raise NotImplementedError
    
class ExplainDataset(HateDataset):
  def __init__(self, config: dict, split: str):
    multis = ["label", "target"]
    unis = ["tokens", "mask", "rationale"]
    super().__init__(config, "explain", split, multis, unis)

  def __getitem__(self, idx):
    num_idx = len(idx)
    if num_idx == 0:
      return {}
    
    label = np.array(self.df.iloc[idx, self.locs["label"]], dtype=np.float32)
    target = np.array(self.df.iloc[idx, self.locs["target"]], dtype=np.float32)

    tokens_list = self.df.iloc[idx, self.locs["tokens"]]
    max_len = max(len(tokens) for tokens in tokens_list)

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
  def __init__(self, config: dict, split: str):
    multis = []
    unis = ["score", "tokens", "mask"]
    super().__init__(config, "measuring", split, multis, unis)
    
  def __getitem__(self, idx):
    num_idx = len(idx)
    if num_idx == 0:
      return {}
    
    score = np.array(self.df.iloc[idx, self.locs["score"]], dtype=np.float32)

    tokens_list = self.df.iloc[idx, self.locs["tokens"]]
    max_len = max(len(tokens) for tokens in tokens_list)

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
  _constructors = {
    "explain": ExplainDataset,
    "measuring": MeasuringDataset,
  }
  _combine_constructors = {
    "dataset": DatasetCombinedDataset,
    "task": TaskCombinedDataset,
  }

  def __init__(self, config, method):
    super().__init__()
    self.config = config
    self.method = method
  
  def _select_data(self, split):
    combiner = self._combine_constructors[self.method]
    dataset = combiner(self.config, {
      name: self._constructors[name](self.config, split)
      for name in self.config["flat_datasets"]
    })
    sampler = BatchSampler(
      RandomSampler(dataset),
      batch_size=self.config["batch_size"],
      drop_last=False
    )
    return dataset, sampler
  
  def setup(self, stage: str):
    self.datasets = {}
    self.samplers = {}
    for split in ["train", "valid", "test"]:
      self.datasets[split], self.samplers[split] = self._select_data(split)

  def _get_dataloader(self, split: str):
    return DataLoader(
      self.datasets[split],
      sampler=self.samplers[split],
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
