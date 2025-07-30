import glob
import pandas as pd

from common import make_config, run_hydra

Config = make_config(
  results_glob = "???",
  confusion_pattern = r"^test_target_confusion_(?P<target>\w+)_(?P<label>\w+)_(?P<pred>\w+)$",
  f1_pattern = r"^test_target_f1_(?P<target>\w+)$",
)

def main(
  results_glob: str,
  confusion_pattern: str,
  f1_pattern: str,
):
  filenames = glob.glob(results_glob)
  df = pd.concat([pd.read_csv(filename) for filename in filenames], axis=1)
  df = df.transpose().reset_index()
  df = df.rename(columns={"index": "name", 0: "value"})

  # micro_f1
  df_extract = df.name.str.extract(confusion_pattern)
  micro_df = pd.concat([df, df_extract], axis=1).dropna()
  micro_df = micro_df.drop(columns=["name", "target"])
  micro_df = micro_df.groupby(["label", "pred"]).sum()

  TP = micro_df.loc[("pos", "pos"), "value"]
  FP = micro_df.loc[("neg", "pos"), "value"]
  FN = micro_df.loc[("pos", "neg"), "value"]
  micro_f1 = 2 * TP / (2 * TP + FP + FN)

  # macro_f1
  df_extract = df.name.str.extract(f1_pattern)
  individual_df = pd.concat([df.drop("name", axis=1), df_extract], axis=1).dropna()
  individual_df = individual_df.sort_values(by=["target"]).reset_index(drop=True) # pyright: ignore
  macro_f1 = individual_df.value.mean()

  # results
  results1_df = individual_df.rename(columns={"target": "name", "value": "f1"})
  results2_df = pd.DataFrame(
    [["micro", micro_f1], ["macro", macro_f1]],
    columns=["name", "f1"], # pyright: ignore
  )
  results_df = pd.concat([results1_df, results2_df]).reindex(columns=["name", "f1"])
  print(results_df.to_string(index=False))

if __name__ == "__main__":
  run_hydra(main, Config)
