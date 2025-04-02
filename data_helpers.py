from multiprocessing import Pool
from tqdm import tqdm
import polars as pl

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
