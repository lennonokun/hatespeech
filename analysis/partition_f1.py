import sys
import glob
import pandas as pd

filenames = glob.glob(sys.argv[1])
df = pd.concat([pd.read_csv(filename) for filename in filenames], axis=1)
df = df.transpose().reset_index()
df = df.rename(columns={"index": "name", 0: "value"})

# micro_f1
df_extract = df.name.str.extract(
  r"^test_target_confusion_(?P<target>\w+)_(?P<label>\w+)_(?P<pred>\w+)$"
)
micro_df = pd.concat([df, df_extract], axis=1).dropna()
micro_df = micro_df.drop(columns=["name", "target"])
micro_df = micro_df.groupby(["label", "pred"]).sum()

TP = micro_df.loc[("pos", "pos"), "value"]
TN = micro_df.loc[("neg", "neg"), "value"]
FP = micro_df.loc[("neg", "pos"), "value"]
FN = micro_df.loc[("pos", "neg"), "value"]
micro_f1 = 2 * TP / (2 * TP + FP + FN)

# macro_f1
target_df = df[df.name.str.contains(r"^test_target_f1_\w+$")]
target_df = target_df.sort_values(by=["name"]).reset_index(drop=True) # pyright: ignore
macro_f1 = target_df.value.mean()

# results
print("PARTITION F1")
print(f"{micro_f1=:.3f}")
print(f"{macro_f1=:.3f}")
print(target_df)
