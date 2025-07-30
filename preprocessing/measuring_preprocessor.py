from pyspark.sql import functions as F

from common import store, fbuilds
from . import utils
from .base_preprocessor import Preprocessor

class MeasuringPreprocessor(Preprocessor):
  name = "measuring"
  
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

preprocessor_store = store(group="preprocessor")
preprocessor_store(fbuilds(
  MeasuringPreprocessor,
  data_info="${data_info}",
  model_name="google/electra-base-discriminator",
  max_length=128,
  round_train=[],
), name="measuring")
