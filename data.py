from transformers import AutoTokenizer
from datasets import Dataset, load_dataset, load_from_disk
from itertools import chain, repeat

from torch.utils.data import DataLoader
from lightning import LightningDataModule

from textattack.transformations import WordSwapRandomCharacterDeletion, \
  WordSwapQWERTY, CompositeTransformation
from textattack.constraints.pre_transformation import \
  RepeatModification, StopwordModification
from textattack.transformations import CompositeTransformation
from textattack.augmentation import Augmenter

class HateAugmenter:
  def __init__(self, config):
    self.config = config
    self.tokenizer = AutoTokenizer.from_pretrained(config["model"])

    transformation = CompositeTransformation([
      WordSwapRandomCharacterDeletion(), WordSwapQWERTY()
    ])
    constraints = [RepeatModification(), StopwordModification()]
    self.augmenter = Augmenter(
      transformation=transformation,
      constraints=constraints,
      pct_words_to_swap=0.3,
      transformations_per_example=config["num_augments"],
    )
    self.dataset = load_dataset("Hate-speech-CNERG/hatexplain")
  
  def _step_majority_label(self, example):
    labels = example["annotators"]["label"]
    example["label"] = max(labels, key=labels.count)
    return example

  def _step_get_text(self, example):
    example["text"] = " ".join(example["post_tokens"])
    return example

  def _step_augment(self, example):
    augmenteds = self.augmenter.augment_many(example["text"])
    return {
      "text": list(chain.from_iterable(
        [text] + aug for text, aug in zip(example["text"], augmenteds)
      )),
      "label": list(chain.from_iterable(
        repeat(label, self.config["num_augments"]+1) for label in example["label"]
      ))
    }

  def augment(self):
    self.dataset = self.dataset.remove_columns(["id", "rationales"])
    self.dataset = self.dataset.map(
      self._step_majority_label,
      remove_columns=["annotators"]
    )
    self.dataset = self.dataset.map(
      self._step_get_text,
      remove_columns=["post_tokens"]
    )
    self.dataset["train"] = self.dataset["train"].map(
      self._step_augment,
      batched=True,
      batch_size=16,
      num_proc=12
    )

  def save(self):
    self.dataset.save_to_disk(self.config["augmented_path"])

class HateData(LightningDataModule):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.tokenizer = AutoTokenizer.from_pretrained(config["model"])

  def _step_tokenize_text(self, example):
    tokenized = self.tokenizer(
      example["text"],
      padding="max_length",
      truncation=True,
      max_length=self.config["max_length"],
    )

    example["tokens"] = tokenized.get("input_ids")
    example["mask"] = tokenized.get("attention_mask")
    return example

  def setup(self, stage: str):
    self.dataset = load_from_disk(self.config["augmented_path"])
    self.dataset = self.dataset.map(
      self._step_tokenize_text,
      batched=True,
      remove_columns=["text"],
    )
    self.dataset = self.dataset.with_format("torch")

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
