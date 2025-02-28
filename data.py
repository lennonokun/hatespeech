from transformers import AutoTokenizer
from datasets import load_dataset

from torch.utils.data import DataLoader
from lightning import LightningDataModule

class HateData(LightningDataModule):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.tokenizer = AutoTokenizer.from_pretrained(config["model"])

  @staticmethod
  def _majority_label(example):
    labels = example["annotators"]["label"]
    example["label"] = max(labels, key=labels.count)
    return example

  def _tokenize_text(self, example):
    texts = [" ".join(text) for text in example["post_tokens"]]
    tokenized = self.tokenizer(
      texts,
      padding="max_length",
      truncation=True,
      max_length=self.config["max_length"],
    )
    example["tokens"] = tokenized.get("input_ids")
    example["mask"] = tokenized.get("attention_mask")
    return example

  def setup(self, stage: str):
    self.dataset = load_dataset("Hate-speech-CNERG/hatexplain")
    self.dataset = self.dataset.map(
      self._majority_label,
      remove_columns=["annotators"]
    )
    self.dataset = self.dataset.map(
      self._tokenize_text,
      batched=True,
      remove_columns=["post_tokens"],
    )
    self.dataset = self.dataset.remove_columns(["id", "rationales"])
    self.dataset = self.dataset.with_format("torch")
    print(self.dataset)

  def _get_dataloader(self, name):
    return DataLoader(
      self.dataset[name],
      batch_size=self.config["batch_size"],
      num_workers=4,
      pin_memory=True
    )

  def train_dataloader(self):
    return self._get_dataloader("train")
  
  def val_dataloader(self):
    return self._get_dataloader("validation")
  
  def test_dataloader(self):
    return self._get_dataloader("test")
