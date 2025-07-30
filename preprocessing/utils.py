from contextlib import contextmanager
from typing import *

from pyspark import SparkContext
from pyspark.sql import (
  SparkSession,
  functions as F,
  types as T
)
import pandas as pd

from transformers import AutoTokenizer

def init_spark_session() -> SparkSession:
  master = SparkSession.builder.master("local[4]") # pyright: ignore[reportAttributeAccessIssue]
  return master \
    .config("spark.sql.execution.arrow.pyspark.enabled", "true") \
    .config("spark.sql.warehouse.dir", "/tmp/spark-warehouse") \
    .config("spark.driver.memory", "2g") \
    .getOrCreate()

def get_spark_session() -> SparkSession:
  return SparkSession.getActiveSession() or init_spark_session()

@contextmanager
def temp_spark_config(inner_values):
  spark = get_spark_session()

  # get and check outer values for Nones
  outer_values = dict()
  for key in inner_values:
    outer_values[key] = spark.conf.get(key, None)

  # set inner values before and set outer after
  for key, inner_value in inner_values.items():
    spark.conf.set(key, inner_value)
  yield
  for key, outer_value in outer_values.items():
    if outer_value is not None:
      spark.conf.set(key, outer_value)

def make_list(x):
  if isinstance(x, list):
    return x
  elif x is None:
    return []
  else:
    return [x]

def prefix_cols(df, cols, prefix):
  for col in cols:
    df = df.withColumnRenamed(col, f"{prefix}{col}")
  return df
  
def segment_bools(ls):
  out = []
  curr_start = None
  for i in range(len(ls)):
    if ls[i] and (i == 0 or not ls[i-1]):
      curr_start = i
    elif not ls[i] and (curr_start is not None):
      out.append((curr_start, i))
      curr_start = None

  if curr_start is not None:
    out.append((curr_start, len(ls) + 1))

  return out

def pudf_tokenize(model_name, max_length):
  sc = SparkContext.getOrCreate()

  tokenizer = AutoTokenizer.from_pretrained(model_name)
  tokenizer = sc.broadcast(tokenizer)

  @F.pandas_udf(T.StructType([ # pyright: ignore[reportPrivateImportUsage, reportCallIssue]
    T.StructField("tokens",     T.ArrayType(T.ShortType())),
    T.StructField("mask",       T.ArrayType(T.ArrayType(T.IntegerType()))),
    T.StructField("offsets",    T.ArrayType(T.ArrayType(T.IntegerType()))),
  ]))
  def func(iterator: Iterator[pd.Series]) -> Iterator[pd.DataFrame]:
    for text in iterator:
      tokenized = tokenizer.value(
        text.to_list(),
        truncation=True,
        max_length=max_length,
        return_offsets_mapping=True,
      )

      yield pd.DataFrame({
        "tokens": tokenized["input_ids"],
        "mask": [segment_bools(x) for x in tokenized["attention_mask"]],
        "offsets": tokenized["offset_mapping"],
      })

  return func

def arr_to_pos_pairs(df, col_id, col_val, sparse=False, sparse_filter=None, include_nulls=False):
  if sparse_filter is not None and not sparse:
    raise ValueError("sparse must be True if sparse_filter is specified.")

  pairs = df.select(col_id, F.posexplode(col_val))
  if sparse_filter is None: sparse_filter = (F.col("col") != 0)
  if sparse: pairs = pairs.filter(sparse_filter)

  if include_nulls:
    pairs = df.select(col_id).join(pairs, col_id, "left")
  return pairs

def pairs_to_map(df, col_id, col_key, col_val, col_out):
  collect_expr = F.collect_list(F.struct(col_key, col_val))
  return df.groupBy(col_id) \
    .agg(F.map_from_entries(collect_expr).alias(col_out))

def map_to_pairs(df, col_id, col_val, include_nulls=False):
  pairs = df.select(col_id, F.explode(F.map_entries(col_val)).alias("entry")) \
    .select(col_id, "entry.key", "entry.value")
  if include_nulls:
    pairs = df.select(col_id).join(pairs, on=col_id, how="left")
  return pairs
    
def empty_map():
  return F.map_from_arrays(F.array(), F.array())

def onehot(df, col_id, col_val, cats, is_arr):
  cat_select = F.array_contains if is_arr else \
    lambda col_val, cat: F.col(col_val) == cat

  return df.select(col_id, *[
    cat_select(col_val, cat).cast("float").alias(cat) for cat in cats
  ])

def apply_cols(df, cols_id, cols_val, expr_func):
  cols_id = make_list(cols_id)
  cols_val = make_list(cols_val)
  return df.select(*cols_id, *[expr_func(col).alias(col) for col in cols_val])

def agg_cols(df, agg, cols_key, cols_val):
  cols_key = make_list(cols_key)
  cols_val = make_list(cols_val)

  return df.groupBy(*cols_key).agg(*[
    agg(F.col(col)).alias(col) for col in cols_val
  ])

def avg_cols(df, cols):
  return df.select(cols) \
    .agg(*[F.avg(col) for col in cols]) \
    .collect()[0]

def avg_map(df, col, keys):
  val_sums = df.select(F.explode(F.map_entries(col)).alias("entry")) \
    .select(F.col("entry.key").alias("key"), F.col("entry.value").alias("value")) \
    .groupBy("key") \
    .agg(F.sum("value").alias("value")) \
    .collect()
  mapping = {row["key"]: row["value"] / df.count() for row in val_sums}
  return [mapping[key] for key in keys]

def sum_all_vals(df, col):
  return df.select(F.explode(F.map_values(col)).alias("value")) \
    .select(F.sum("value")) \
    .collect()[0][0]

def sum_span_widths(df, col):
  return df.select(F.explode(col).alias(col)) \
    .select((F.col(col).getItem(1) - F.col(col).getItem(0)).alias("width")) \
    .agg(F.sum("width")) \
    .collect()[0][0]

# doesn't work with null values
def batched_pdf(df, pudf, col_key, col_args, batch_size):
  with temp_spark_config({"spark.sql.execution.arrow.maxRecordsPerBatch": batch_size}):
    results = df.select(col_key, pudf(*col_args).alias("ret"))
    return results.select(col_key, "ret.*")
