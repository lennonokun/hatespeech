import polars as pl
import numpy as np
from tqdm import tqdm

from multiprocessing import Pool

from torch.utils.data import TensorDataset, DataLoader, random_split
from lightning import LightningDataModule

from textattack.augmentation.recipes import EasyDataAugmenter
from nltk.tokenize.treebank import TreebankWordDetokenizer
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
    self.detokenizer = TreebankWordDetokenizer()
    self.augmenter = EasyDataAugmenter(transformations_per_example=config["num_augments"])

    self.df = pl.read_json("data/dataset.json")
    self.df = self.process_data(self.df)
    print(self.df)
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

  def _augment_batch(self, batch):
    augmenteds = self.augmenter.augment_many(batch["text"])
    all_texts = [[text, *augmented] for text, augmented in zip(batch["text"], augmenteds)]
    return {"texts": all_texts}

  def detokenize(self, tokens):
    text = self.detokenizer.detokenize(tokens)
    spans = []
    curr_start = 0

    for token in tokens:
      if (found := text.find(token, curr_start)) != -1:
        spans.append((found, found+len(token)))
        curr_start = found + len(token)
      elif (found := text.find(token.replace(" ", ""), curr_start)) != -1:
        spans.append((found, found+len(token)))
        curr_start = found + len(token)
      else:
        spans.append((-1, -1))
    return {"text": text, "spans": spans}
  
  def augment(self, df):
    df_train = df.filter(pl.col("split") == "train")
    df_other = df.filter(pl.col("split") != "train")

    if self.config["do_augment"]:
      df_train = multi_batched_map(
        df_train,
        self._augment_batch,
        {"texts": pl.List(pl.String)},
        self.config["augment_batch_size"],
        self.config["augment_num_workers"],
      ).drop("text")
    else:
      df_train = df_train.with_columns(
        pl.col("text").map_elements(lambda x: [x], return_dtype=pl.List(pl.String)).alias("texts")
      ).drop("text")
    df_other = df_other.with_columns(
      pl.col("text").map_elements(lambda x: [x], return_dtype=pl.List(pl.String)).alias("texts")
    ).drop("text")

    return pl.concat([df_train, df_other], how="vertical")

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

    # detokenize post tokens
    df = df.with_columns(
      pl.col("post_tokens").map_elements(
        self.detokenize, return_dtype=pl.Struct([
          pl.Field("text", pl.String),
          pl.Field("spans", pl.List(pl.List(pl.Int64))),
        ]),
      ).alias("detokenized")
    ).drop("post_tokens").unnest("detokenized")

    # add random train val test split
    split_fracs = np.random.permutation(len(df)) / len(df)
    split_conds = [split_fracs < 0.8, split_fracs < 0.9]
    split_vals = np.select(split_conds, ["train", "valid"], default="test")
    df = df.with_columns(pl.Series("split", split_vals, dtype=pl.Categorical()))
    
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
      .drop(["index", "null"]) \
      .select(pl.all().name.prefix("target_"))
    df = df.drop("target").to_dummies("label")
    df = pl.concat([df, df_target], how="horizontal")

    # aggregate with mean rationales, labels, and targets
    df = df.group_by("post_id").agg([
      pl.col("rationales").map_elements(
        self._udf_rationale_mean,
        return_dtype=pl.List(pl.Float32)
      ),
      pl.col("text").first(),
      pl.col("split").first(),
      pl.col("spans").first(),
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
      return_offsets_mapping=True,
    )

    # find token labels
    tokens = np.array(tokenized["input_ids"], dtype=np.int32)
    mask = np.array(tokenized["attention_mask"], dtype=np.int32)
    offsets = np.array(tokenized["offset_mapping"], dtype=np.int32)
    spans = rows["spans"]
    rationales = rows["rationales"]

    rationales2 = np.zeros((len(tokens), self.config["max_length"]) , dtype=np.float32)

    for i in range(len(rows)):
      for rationale, (span_start, span_end) in zip(rationales[i], spans[i]):
        if rationale > 0:
          for j in range(self.config["max_length"]):
            check_left = offsets[i, j, 0] >= span_start
            check_right = offsets[i, j, 1] <= span_end
            valid = offsets[i, j, 0] != offsets[i, j, 1]
            if check_left and check_right and valid:
              rationales2[i, j] = rationale

    return {
      "tokens": tokens.tolist(),
      "mask": mask.tolist(),
      "rationales2": rationales2.tolist()
    }

  def setup(self, stage: str):
    df = pl.read_parquet(self.config["augmented_path"])
    df = df.explode("texts")
    df = multi_batched_map(
      df, self._tokenize_batch,
      {
        "tokens": pl.Array(pl.Int64, self.config["max_length"]),
        "mask": pl.Array(pl.Int64, self.config["max_length"]),
        "rationales2": pl.Array(pl.Float32, self.config["max_length"])
      },
      self.config["tokenize_batch_size"],
    ).drop(["texts", "rationales"]).rename({"rationales2": "rationales"})
    print(df)
    # df = df.group_by("post_id").agg(pl.col("tokens", "mask", "label"), pl.col("split").first())

    self.datasets = {}
    for num_augments, split in zip([self.config["num_augments"]+1, 1, 1], ["train", "valid", "test"]):
      # df_split = df.filter(pl.col("split") == split).filter(pl.col("tokens").list.len() == num_augments).select(
      df_split = df.filter(pl.col("split") == split).select(
        pl.col("tokens").cast(pl.Array(pl.Int64, self.config["max_length"])),
        pl.col("mask").cast(pl.Array(pl.Int64, self.config["max_length"])),
        pl.col("label").cast(pl.Array(pl.Float32, self.config["num_labels"])),
        pl.col("target").cast(pl.Array(pl.Float32, self.config["num_targets"])),
        pl.col("rationales").cast(pl.Array(pl.Float32, self.config["max_length"])),
      )
      self.datasets[split] = TensorDataset(*[
        df_split[feature].to_torch() for feature in ["tokens", "mask", "label", "target", "rationales"]
      ])
    
  def _get_dataloader(self, split):
    return DataLoader(
      self.datasets[split],
      batch_size=self.config["batch_size"],
      num_workers=4,
      pin_memory=True
    )

  def train_dataloader(self):
    return self._get_dataloader("train")
  
  def val_dataloader(self):
    return self._get_dataloader("valid")
  
  def test_dataloader(self):
    return self._get_dataloader("test")
