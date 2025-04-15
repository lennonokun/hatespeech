from typing import *

import json
import numpy as np

from pyspark import SparkContext
from pyspark.sql import functions as F, types as T

from nltk.tokenize.treebank import TreebankWordDetokenizer

from . import utils
from .preprocessor import Preprocessor

schema = T.StructType([
    T.StructField("post_id", T.StringType(), True),
    T.StructField("annotators", T.ArrayType(
        T.StructType([
            T.StructField("label", T.StringType(), True),
            T.StructField("annotator_id", T.IntegerType(), True),
            T.StructField("target", T.ArrayType(T.StringType()), True)
        ])
    ), True),
    T.StructField("rationales", T.ArrayType(T.ArrayType(T.IntegerType()))),
    T.StructField("post_tokens", T.ArrayType(T.StringType()), True)
])

def do_fix(config):
  # read from json
  with open(config["explain_dirty_path"], "r") as f:
    data = json.load(f)

  # get in spark
  spark = utils.get_spark_session()
  sc = SparkContext.getOrCreate()
  rdd = sc.parallelize(data.values(), numSlices=10)
  df = spark.createDataFrame(rdd, schema)

  # write to parquet
  path = config["input_dataset_path"].format(name="explain")
  df.write.mode("overwrite").parquet(path)

class ExplainPreprocessor(Preprocessor):
  def __init__(self, config):
    super().__init__(config, "explain")

    self.detokenizer = TreebankWordDetokenizer()

    self.udf_detokenize = self._udf_detokenize(self.detokenizer)
    self.pudf_tokenize = utils.pudf_tokenize(config)

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
    
  def preprocess(self, df):
    df = df.withColumn("id", F.monotonically_increasing_id()) \
      .drop("post_id")

    print("do rationale")
    df_rationale = df.select("id", F.explode("rationales").alias("rationale"))
    df_rationale = utils.arr_to_pos_pairs(df_rationale, "id", "rationale", sparse=True)
    df_rationale = utils.agg_cols(df_rationale, F.sum, ["id", "pos"], "col") \
      .withColumn("col", F.col("col").cast("float") / self.config["num_annotators"])
    if "rationale" in self.config["round_train"]:
      df_rationale = df_rationale.withColumn("col", F.round("col"))
    df_rationale = utils.pairs_to_map(df_rationale, "id", "pos", "col", "rationale")
    df = utils.drop_join(df, df_rationale, "id", "rationales") \
      .withColumn("rationale", F.coalesce("rationale", utils.empty_map()))
    
    df.show()

    print("do label + target")
    df_annotator = df.select("id", F.explode("annotators").alias("annotator"))
    df_label = utils.cat_to_onehot(df_annotator, "id", "annotator.label", self.config["labels"]) \
      .groupBy("id") \
      .agg(*[(F.sum(F.col(cat).cast("float")) / self.config["num_annotators"]).alias(f"label_{cat}") for cat in self.config["labels"]])
    if "label" in self.config["round_train"]:
      cols = [f"label_{val}" for val in self.config["labels"]]
      df_label = df_label.select("id", *[F.round(col).alias(col) for col in cols])
    
    df_label.show()
    df_target = df_annotator.select("id", F.explode("annotator.target").alias("target"))
    df_target = utils.cat_to_ord(df_target, "id", "target", self.config["targets"])  \
      .filter(F.col("target").isNotNull()) \
      .groupBy("id", "target") \
      .count() \
      .withColumn("count", F.col("count").cast("float") / self.config["num_annotators"])
    if "target" in self.config["round_train"]:
      df_target = df_target.withColumn("count", F.round("count"))
    df_target = utils.pairs_to_map(df_target, "id", "target", "count", col_out="target")
    df_target.show()
    
    df = utils.drop_join(df, df_label, "id", "annotators")
    df = utils.drop_join(df, df_target, "id") \
      .withColumn("target", F.coalesce("target", utils.empty_map()))

    df.show()

    # detokenize and get text + original spans
    print("do detokenize")
    df_detokenize = df.select("id", self.udf_detokenize("post_tokens").alias("ret")) \
      .select("id", "ret.*")
    df = utils.drop_join(df, df_detokenize, "id", "post_tokens")

    # retokenize and adjust rationale with original spans
    print("do tokenize")
    df_tokenized = utils.batched_pdf(df, self.pudf_tokenize, "id", ["text", "rationale", "spans"], 256)
    df = utils.drop_join(df, df_tokenized, "id", ["text", "rationale", "spans"])

    df.show()
    return df

  def get_stats(self, df):
    get_freqs = lambda avgs: [avg / self.config["num_annotators"] for avg in avgs]
    get_freq = lambda avg: avg / self.config["num_annotators"]
    
    label_cols = [f"label_{val}" for val in self.config["labels"]]
    label_avgs = utils.avg_cols(df, label_cols)
    target_avgs = utils.avg_map(df, "target", range(self.config["num_targets"]))
    rationale_sum = utils.sum_all_vals(df, "rationale")
    mask_sum = utils.sum_span_widths(df, "mask")

    return {
      "label_freqs": get_freqs(label_avgs),
      "target_freqs": get_freqs(target_avgs),
      "rationale_freq": get_freq(rationale_sum / mask_sum)
    }
