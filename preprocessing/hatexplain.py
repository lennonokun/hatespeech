import json
import numpy as np
import pandas as pd

from pyspark import SparkContext
from pyspark.sql import functions as F, types as T

from typing import *
from nltk.tokenize.treebank import TreebankWordDetokenizer
from transformers import AutoTokenizer

from . import helpers as help

# fix bad nested format
def do_fix(config):
  with open(config["dirty_dataset_path"], "r") as f:
    data = json.load(f)
    data = list(data.values())
  with open(config["clean_dataset_path"], "w") as f:
    f.write(json.dumps(data))

class HatePreprocessor:
  def __init__(self, config):
    self.config = config
    self.detokenizer = TreebankWordDetokenizer()
    self.tokenizer = AutoTokenizer.from_pretrained(config["model"])

    sc = SparkContext.getOrCreate()
    self.broadcast_tokenizer = sc.broadcast(self.tokenizer)
    
    self.udf_detokenize = self._udf_detokenize(self.detokenizer)
    self.udf_batch_tokenize = self._udf_batch_tokenize(self.broadcast_tokenizer, config)

  @staticmethod
  @F.udf(returnType=T.ArrayType(T.FloatType(), False))
  def udf_rationale_mean(rationale):
    rationale = [r for r in rationale if r is not None]
    if len(rationale) == 0:
      return []
    if len(set(len(r) for r in rationale)) > 1:
      return []
    else:
      return np.mean(rationale, axis=0).tolist()

  @staticmethod
  def _udf_detokenize(detokenizer):
    @F.udf(returnType=T.StructType([
      T.StructField("text", T.StringType(), False),
      T.StructField("spans", T.ArrayType(T.ArrayType(T.IntegerType(), False), False), False)
    ]))
    def func(tokens):
      text = detokenizer.detokenize(tokens)
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

      return text, spans

    return func

  def load_prev_test(self):
    spark = help.get_spark_session()
    print("do read")
    self.df = spark.read.parquet(self.config["preprocessed_dataset_paths"]["train"])
    self.df.show()
    
  @staticmethod
  def _udf_batch_tokenize(broadcast_tokenizer, config):
    @F.pandas_udf(T.StructType([ # pyright: ignore[reportPrivateImportUsage, reportCallIssue]
      T.StructField("tokens",     T.ArrayType(T.ShortType())),
      T.StructField("mask",       T.ArrayType(T.ArrayType(T.ShortType()))),
      # T.StructField("offsets",    T.ArrayType(T.ArrayType(T.IntegerType()))),
      T.StructField("rationale", T.MapType(T.IntegerType(), T.FloatType()))
    ]))
    def func(iterator: Iterator[Tuple[pd.Series, pd.Series, pd.Series]]) -> Iterator[pd.DataFrame]:
      for text, rationale, spans in iterator:
        text, rationale, spans = tuple(x.to_list() for x in [text, rationale, spans])
        
        tokenized = broadcast_tokenizer.value(
          text,
          # padding="max_length",
          truncation=True,
          max_length=config["max_length"],
          return_offsets_mapping=True,
        )

        masks = [help.segment_bools(mask) for mask in tokenized["attention_mask"]]
        offsets = tokenized["offset_mapping"]
        rationale2 = [dict() for _ in range(len(text))]

        if rationale is None:
          print("rationale is none")
          
        if rationale is not None:
          for i in range(len(text)):
            if rationale[i] is not None:
              for (r_key, r_val) in rationale[i].items():
                if r_key < len(spans[i]) and r_val > 0:
                  s_start, s_end = tuple(spans[i][r_key])
                  for j in range(len(offsets[i])):
                    check_left = offsets[i][j][0] >= s_start
                    check_right = offsets[i][j][1] <= s_end
                    valid = offsets[i][j][0] != offsets[i][j][1]
                    if check_left and check_right and valid:
                      rationale2[i][j] = r_val

        yield pd.DataFrame({
          "tokens": tokenized["input_ids"],
          "mask": masks,
          "rationale": rationale2,
          # "offsets": offsets.tolist(),
        })

    return func
  
  def preprocess(self):
    spark = help.get_spark_session()
    print("do read")
    df = spark.read.option("multiline", "true").json(self.config["clean_dataset_path"])
    df = df.withColumn("id", F.monotonically_increasing_id()) \
      .drop("post_id")

    df.show()

    print("do rationale")
    df_rationale = df.select("id", F.explode("rationales").alias("rationale"))
    df_rationale = help.arr_to_pos_pairs(df_rationale, "id", "rationale", sparse=True)
    df_rationale = help.agg_cols(df_rationale, F.sum, ["id", "pos"], "col") \
      .withColumn("col", F.col("col").cast("float") / self.config["num_annotators"])
    if "rationale" in self.config["round_train"]:
      df_rationale = df_rationale.withColumn("col", F.round("col"))
    df_rationale = help.pairs_to_map(df_rationale, "id", "pos", "col", "rationale")
    df = help.drop_join(df, df_rationale, "id", "rationales") \
      .withColumn("rationale", F.coalesce("rationale", help.empty_map()))
    
    df.show()

    print("do label + target")
    df_annotator = df.select("id", F.explode("annotators").alias("annotator"))
    df_label = help.cat_to_onehot(df_annotator, "id", "annotator.label", self.config["labels"]) \
      .groupBy("id") \
      .agg(*[(F.sum(F.col(cat).cast("float")) / self.config["num_annotators"]).alias(f"label_{cat}") for cat in self.config["labels"]])
    if "label" in self.config["round_train"]:
      cols = [f"label_{val}" for val in self.config["labels"]]
      df_label = df_label.select("id", *[F.round(col).alias(col) for col in cols])
    
    df_label.show()
    df_target = df_annotator.select("id", F.explode("annotator.target").alias("target"))
    df_target = help.cat_to_ord(df_target, "id", "target", self.config["targets"])  \
      .filter(F.col("target").isNotNull()) \
      .groupBy("id", "target") \
      .count() \
      .withColumn("count", F.col("count").cast("float") / self.config["num_annotators"])
    if "target" in self.config["round_train"]:
      df_target = df_target.withColumn("count", F.round("count"))
    df_target = help.pairs_to_map(df_target, "id", "target", "count", col_out="target")
    df_target.show()
    
    df = help.drop_join(df, df_label, "id", "annotators")
    df = help.drop_join(df, df_target, "id") \
      .withColumn("target", F.coalesce("target", help.empty_map()))

    df.show()

    # detokenize and get text + original spans
    print("do detokenize")
    df_detokenize = df.select("id", self.udf_detokenize("post_tokens").alias("ret")) \
      .select("id", "ret.*")
    df = help.drop_join(df, df_detokenize, "id", "post_tokens")

    # retokenize and adjust rationale with original spans
    print("do tokenize")
    df_tokenized = help.batched_pdf(df, self.udf_batch_tokenize, "id", ["text", "rationale", "spans"], 256)
    df = help.drop_join(df, df_tokenized, "id", ["text", "rationale", "spans"])

    df.show()

    self.df = df

    # apply rounding
    # print("do rounding")
    # # if "label" in self.confi["round_train"]:
    # #   df = df.withColumn(label, F.
    # for col in ["target", "rationale"]:
    #   if col in self.config["round_train"]:
    #     df = df.withColumn(col, F.expr(f"transform_values({col}, (k,v) -> (k, round(v)))"))
    # self.df = df

  def get_stats(self):
    get_freqs = lambda avgs: [avg / self.config["num_annotators"] for avg in avgs]
    get_freq = lambda avg: avg / self.config["num_annotators"]
    
    label_cols = [f"label_{val}" for val in self.config["labels"]]
    label_avgs = help.avg_cols(self.df, label_cols)
    target_avgs = help.avg_map(self.df, "target", range(self.config["num_targets"]))
    rationale_sum = help.sum_all_vals(self.df, "rationale")
    mask_sum = help.sum_span_widths(self.df, "mask")

    return {
      "label_freqs": get_freqs(label_avgs),
      "target_freqs": get_freqs(target_avgs),
      "rationale_freq": get_freq(rationale_sum / mask_sum)
    }

  def write(self):
    print("do shuffle")
    self.df = self.df.orderBy(F.rand())
    print("do split")
    split_dfs = self.df.randomSplit([0.8, 0.1, 0.1])
    print("do write")
    split_names = ["train", "valid", "test"]
    for split_df, split_name in zip(split_dfs, split_names):
      file_name = self.config["preprocessed_dataset_paths"][split_name]
      split_df.write.mode("overwrite").parquet(file_name)
    
    stats = self.get_stats()
    with open(self.config["stats_path"], "w") as f: 
      f.write(json.dumps(stats))
