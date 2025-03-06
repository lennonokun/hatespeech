import polars as pl
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, random_split
from lightning import LightningDataModule

from textattack.transformations import WordSwapRandomCharacterDeletion, \
  WordSwapQWERTY, CompositeTransformation
from textattack.constraints.pre_transformation import \
  RepeatModification, StopwordModification
from textattack.augmentation import Augmenter
from transformers import AutoTokenizer

def multi_batched_map(df, func, schema, batch_size, num_workers=1):
  batched = df.with_columns(batch_idx=pl.int_range(0, len(df)) // batch_size) \
    .group_by("batch_idx").agg(pl.all()) \
    .sort("batch_idx").drop("batch_idx")

  if num_workers == 1:
    batched2 = pl.from_dicts(iter(tqdm(
      (func(batch) for batch in batched.iter_rows(named=True)),
      total=len(batched),
    )), {k: pl.List(v) for k,v in schema.items()})
  else:
    with Pool(num_workers) as pool:
      batched2 = pl.from_dicts(iter(tqdm(
        pool.imap(func, batched.iter_rows(named=True)),
        total=len(batched),
      )), {k: pl.List(v) for k,v in schema.items()})

  return pl.concat([df, batched2.explode(pl.all())], how="horizontal")

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
    self.df = self.process_data(pl.read_json("data/dataset.json"))
    self.df = self.augment(self.df)
    self.df.write_parquet(self.config["augmented_path"])

  @staticmethod
  def _udf_rationale_mean(rationales):
    rationales = [r for r in rationales.to_list() if r is not None]
    if len(rationales) == 0:
      return None
    if len(set(len(r) for r in rationales)) > 1:
      return None
    else:
      return np.mean(rationales, axis=0)

  def process_data(self, df):
    # make into record format
    df = df.transpose(column_names=["struct"]) \
      .unnest("struct").with_columns(
        pl.col("annotators").cast(pl.List(pl.Struct([
          pl.Field("label", pl.Categorical),
          pl.Field("annotator_id", pl.UInt16),
          pl.Field("target", pl.List(pl.Categorical)),
        ]))),
        pl.col("rationales").cast(pl.List(pl.List(pl.UInt8))),
      )

    # explode annotators and padded rationales
    df = df.with_columns(
        pl.col("rationales").list.gather(
          pl.int_range(self.config["num_annotators"]),
          null_on_oob=True
        ).fill_null([])
      ).explode("annotators", "rationales") \
      .unnest("annotators")

    # make label and target one-hot
    df_target = df.with_columns(pl.lit(1).alias("one")) \
      .with_row_index("index") \
      .explode("target") \
      .pivot(on="target", index="index", values="one", aggregate_function="first") \
      .fill_null(0) \
      .drop(["null", "index"]) \
      .select(pl.all().name.prefix("target_"))
    df = df.drop("target").to_dummies("label")
    df = pl.concat([df, df_target], how="horizontal")

    # aggregate with mean rationales, labels, and targets
    df = df.group_by("post_id").agg([
      pl.col("rationales").map_elements(
        self._udf_rationale_mean,
        return_dtype=pl.List(pl.Float32)
      ),
      pl.col("post_tokens").first(),
      pl.col("^label_.*$").cast(pl.Float32).mean(),
      pl.col("^target_.*$").cast(pl.Float32).mean(),
    ])

    # combine labels and targets into one array
    label_names = [f"label_{name}" for name in self.config["labels"]]
    target_names = [f"target_{name}" for name in self.config["targets"]]
    df = df.with_columns(
      label=pl.concat_arr(label_names).cast(pl.Array(pl.Float32, self.config["num_labels"])),
      target=pl.concat_arr(target_names).cast(pl.Array(pl.Float32, self.config["num_targets"])),
    ).drop(label_names + target_names)

    return df

  def _augment_batch(self, batch):
    texts = batch["post_tokens"]
    augmenteds = self.augmenter.augment_many(texts)
    all_texts = [[text, *augmented] for text, augmented in zip(texts, augmenteds)]
    return {"texts": all_texts}
  
  def augment(self, df):
    df = df.with_columns(pl.col("post_tokens").list.join(" "))
    return multi_batched_map(
      df,
      self._augment_batch,
      {"texts": pl.Array(pl.String, self.config["num_augments"] + 1)},
      self.config["augment_batch_size"],
      self.config["augment_num_workers"],
    ).drop("post_tokens")

class HateData(LightningDataModule):
  def __init__(self, config):
    super().__init__()
    self.config = config
    self.tokenizer = AutoTokenizer.from_pretrained(config["model"])

  def _tokenize_batch(self, rows):
    tokenized = self.tokenizer(
      rows["texts"],
      padding="max_length",
      truncation=True,
      max_length=self.config["max_length"],
    )
    return {
      "tokens": tokenized["input_ids"],
      "mask": tokenized["attention_mask"],
    }

  def setup(self, stage: str):
    df = pl.read_parquet(self.config["augmented_path"])
    df = df.explode("texts")
    df = multi_batched_map(
      df, self._tokenize_batch,
      {
        "tokens": pl.Array(pl.Int64, self.config["max_length"]),
        "mask": pl.Array(pl.Int64, self.config["max_length"])
      },
      self.config["tokenize_batch_size"],
    ).drop("texts")
    df = df.group_by("post_id").agg(pl.col("tokens", "mask", "label"))
    df = df.select(
      pl.col("tokens").cast(pl.Array(pl.Array(pl.Int64, self.config["max_length"]), self.config["num_augments"]+1)),
      pl.col("mask").cast(pl.Array(pl.Array(pl.Int64, self.config["max_length"]), self.config["num_augments"]+1)),
      pl.col("label").cast(pl.Array(pl.Array(pl.Float32, self.config["num_labels"]), self.config["num_augments"]+1)),
    )

    dataset = TensorDataset(df["tokens"].to_torch(), df["mask"].to_torch(), df["label"].to_torch())
    datasets = random_split(dataset, [0.8, 0.1, 0.1])
    splits = ["train", "val", "test"]
    self.datasets = {k: v for k,v in zip(splits, datasets)}

  def _get_dataloader(self, name):
    return DataLoader(
      self.datasets[name],
      batch_size=self.config["batch_size"],
      num_workers=4,
      pin_memory=True
    )

  def train_dataloader(self):
    return self._get_dataloader("train")
  
  def val_dataloader(self):
    return self._get_dataloader("val")
  
  def test_dataloader(self):
    return self._get_dataloader("test")
