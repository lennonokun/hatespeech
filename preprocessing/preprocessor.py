import json

from pyspark.sql import functions as F

from . import utils

class Preprocessor:
  def __init__(self, config, name):
    self.config = config
    self.name = name

  def preprocess(self, df):
    raise NotImplementedError

  def get_stats(self, df):
    raise NotImplementedError

  def execute(self):
    df = self.preprocess(self.read())
    stats = self.get_stats(df)
    self.write(df, stats)

  def read(self):
    print("do read")
    spark = utils.get_spark_session()
    path = self.config["input_dataset_path"].format(name=self.name)
    return spark.read.parquet(path)

  def write(self, df, stats):
    print("do write")
    split_dfs = df.orderBy(F.rand()).randomSplit([0.8, 0.1, 0.1])
    split_names = ["train", "valid", "test"]
    for split_df, split in zip(split_dfs, split_names):
      file_path = self.config["output_dataset_path"] \
        .format(name=self.name, split=split)
      split_df.write.mode("overwrite").parquet(file_path)
    
    file_path = self.config["output_stats_path"].format(name=self.name)
    with open(file_path, "w") as f:
      f.write(json.dumps(stats))
