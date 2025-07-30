import json
from pyspark import SparkContext
from pyspark.sql import functions as F, types as T
from nltk.tokenize.treebank import TreebankWordDetokenizer

from common import DataInfo, fbuilds, store
from . import utils
from .base_preprocessor import Preprocessor

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

def fix_explain(data_info: DataInfo):
  # read from json
  print("reading from json")
  with open(data_info.explain_dirty_path, "r") as f:
    data = json.load(f)

  # get in spark
  print("loading into spark")
  spark = utils.get_spark_session()
  sc = SparkContext.getOrCreate()
  rdd = sc.parallelize(data.values(), numSlices=10)
  df = spark.createDataFrame(rdd, schema)

  # write to parquet
  print("writing to parquet")
  path = data_info.input_dataset_path.format(name="explain")
  df.write.mode("overwrite").parquet(path)

  print("done!")

class ExplainPreprocessor(Preprocessor):
  name = "explain"
  
  def _post_init(self):
    self.detokenizer = TreebankWordDetokenizer()
    self.udf_detokenize = self._udf_detokenize(self.detokenizer)

  @staticmethod
  def _udf_detokenize(detokenizer):
    @F.udf(returnType=T.StructType([
      T.StructField("text", T.StringType(), False),
      T.StructField("spans", T.ArrayType(T.ArrayType(T.IntegerType())))
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

  @staticmethod
  @F.udf(returnType=T.MapType(T.IntegerType(), T.FloatType()))
  def udf_adjust_rationale(rationale, spans, offsets):
    adjusted = {}
    for new_pos, (new_start, new_end) in enumerate(offsets):
      for old_pos, (old_start, old_end) in enumerate(spans):
        if (new_start >= old_start) and (new_end <= old_end):
          if old_pos in rationale:
            adjusted[new_pos] = rationale[old_pos]
      return adjusted

  def preprocess(self, df):
    df = df.withColumn("id", F.monotonically_increasing_id()).drop("post_id")

    # label + target
    df_annotator = df.select("id", F.explode("annotators").alias("annotator")) \
      .select("id", *[F.col(f"annotator.{member}") for member in ["label", "target"]]) 

    df_label = utils.onehot(df_annotator, "id", "label", self.data_info.cats["label"], False)
    df_label = utils.agg_cols(df_label, F.avg, "id", self.data_info.cats["label"])
    df_label = utils.prefix_cols(df_label, self.data_info.cats["label"], "label_")
    if "label" in self.round_train:
      df_label = utils.apply_cols(df_label, "id", self.data_info.cols["label"], F.round)
    
    df_target = utils.onehot(df_annotator, "id", "target", self.data_info.cats["target"], True)
    df_target = utils.agg_cols(df_target, F.avg, "id", self.data_info.cats["target"])
    df_target = utils.prefix_cols(df_target, self.data_info.cats["target"], "target_")
    if "target" in self.round_train:
      df_target = utils.apply_cols(df_target, "id", self.data_info.cols["target"], F.round)
    
    df = df.drop("annotators").join(df_label, "id", "left").join(df_target, "id", "left")
    
    # rationale agg + pos map 
    df_rationale = df.select("id", F.explode("rationales").alias("rationale"))
    df_rationale = utils.arr_to_pos_pairs(df_rationale, "id", "rationale", sparse=True)
    df_rationale = utils.agg_cols(df_rationale, F.avg, ["id", "pos"], "col")
    if "rationale" in self.round_train:
      df_rationale = df_rationale.withColumn("col", F.round("col"))
    df_rationale = utils.pairs_to_map(df_rationale, "id", "pos", "col", "rationale")
    df = df.drop("rationales").join(df_rationale, "id", "left") \
      .withColumn("rationale", F.coalesce("rationale", utils.empty_map()))
    
    # detokenize
    df_detokenize = df.select("id", self.udf_detokenize("post_tokens").alias("ret")) \
      .select("id", "ret.*")
    df = df.drop("post_tokens").join(df_detokenize, "id")

    # tokenize
    df_tokenize = utils.batched_pdf(df, self.pudf_tokenize, "id", ["text"], 256)
    df = df.drop("text").join(df_tokenize, "id")

    # adjust rationales
    adjust_expr = self.udf_adjust_rationale("rationale", "spans", "offsets")
    df = df.withColumn("rationale2", adjust_expr) \
      .drop("rationale", "spans", "offsets") \
      .withColumnRenamed("rationale2", "rationale")
   
    return df

  def get_stats(self, df):
    label_avgs = utils.avg_cols(df, self.data_info.cols["label"])
    target_avgs = utils.avg_cols(df, self.data_info.cols["target"])
    rationale_sum = utils.sum_all_vals(df, "rationale")
    mask_sum = utils.sum_span_widths(df, "mask")

    return {
      "label_freqs": label_avgs,
      "target_freqs": target_avgs,
      "rationale_freq": rationale_sum / mask_sum
    }

preprocessor_store = store(group="preprocessor")
preprocessor_store(fbuilds(
  ExplainPreprocessor,
  data_info="${data_info}",
  model_name="google/electra-base-discriminator",
  max_length=128,
  round_train=["target", "label", "rationale"],
), name="explain")
