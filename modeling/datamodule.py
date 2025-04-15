import numpy as np
import json
from lightning import LightningDataModule

import torch
from torch.utils.data import Dataset, DataLoader
import pyarrow.parquet as pq

# not scalable, but the dataset is small enough
class HateDataset(Dataset):
  def __init__(self, config: dict, file_path: str):
    self.config = config
    self.df = pq.ParquetDataset(file_path).read().to_pandas()
    self.label_cols = [f"label_{name}" for name in self.config["labels"]]

  def __len__(self):
    return len(self.df)

  def __getitem__(self, idx):
    row = self.df.iloc[idx]

    label = np.array(row[self.label_cols], dtype=np.float32)

    target = np.zeros(self.config["num_targets"], dtype=np.float32)
    for (pos, val) in row["target"]:
      target[pos] = val

    tokens = np.array(row["tokens"], dtype=np.int32)

    mask = np.zeros(self.config["max_length"], dtype=np.bool_)
    for (start, end) in row["mask"]:
      mask[start: end] = True
    
    rationale = np.zeros(self.config["max_length"], dtype=np.float32)
    for (pos, val) in row["rationale"]:
      rationale[pos] = val

    return {
      "tokens": tokens,
      "mask": mask,
      "label": label,
      "target": target,
      "rationale": rationale,
    }

class HateDatamodule(LightningDataModule):
  def __init__(self, config):
    super().__init__()
    self.config = config

  def setup(self, stage: str):
    path = self.config["output_stats_path"].format(name="explain")
    self.stats = json.load(open(path, "r"))

    self.datasets = {}
    for split in ["train", "valid", "test"]:
      path = self.config["output_dataset_path"].format(name="explain", split=split)
      self.datasets[split] = HateDataset(self.config, path)

  def _collate_fn(self, samples):
    max_length = max(len(sample["tokens"]) for sample in samples)

    tokens_batch = np.zeros((len(samples), max_length), dtype=np.int32)
    for i, sample in enumerate(samples):
      tokens_batch[i, :len(sample["tokens"])] = sample["tokens"]
    
    mask_batch = np.vstack(tuple(sample["mask"] for sample in samples))
    rationale_batch = np.vstack(tuple(sample["rationale"] for sample in samples))
    mask_batch = mask_batch[:, :max_length]
    rationale_batch = rationale_batch[:, :max_length]

    label_batch = np.vstack(tuple(sample["label"] for sample in samples))
    target_batch = np.vstack(tuple(sample["target"] for sample in samples))

    return (
      torch.IntTensor(tokens_batch),
      torch.BoolTensor(mask_batch),
      torch.FloatTensor(label_batch),
      torch.FloatTensor(target_batch),
      torch.FloatTensor(rationale_batch),
    )

  def _get_dataloader(self, split: str):
    # return DataLoader(self.datasets[split])
    return DataLoader(
      self.datasets[split],
      batch_size=self.config["batch_size"],
      num_workers=4,
      pin_memory=True,
      collate_fn=self._collate_fn,
    )

  def train_dataloader(self):
    return self._get_dataloader("train")
  
  def val_dataloader(self):
    return self._get_dataloader("valid")
  
  def test_dataloader(self):
    return self._get_dataloader("test")
