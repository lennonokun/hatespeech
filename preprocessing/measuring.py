from pyspark.sql import functions as F

from . import utils
from .preprocessor import Preprocessor

class MeasuringPreprocessor(Preprocessor):
  name = "measuring"
  
  def __init__(self, config):
    super().__init__(config)
    self.config = config
    self.pudf_tokenize = utils.pudf_tokenize(config)
  
  def preprocess(self, df):
    df = df.groupBy("comment_id") \
      .agg(F.avg("hate_speech_score").alias("score"), F.first("text").alias("text"))

    df_tokenized = utils.batched_pdf(df, self.pudf_tokenize, "comment_id", ["text"], 256)
    df = df.join(df_tokenized, "comment_id") \
      .drop("text", "rationale", "spans")

    df.show()
    return df

  def get_stats(self, df):
    return {}
