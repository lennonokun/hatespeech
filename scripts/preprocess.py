from preprocessing import Preprocessor
from common import DataInfoCfg, run_hydra, make_config

Config = make_config(
  hydra_defaults=[
    "_self_",
    {"preprocessor": "???"},
  ],
  preprocessor=None,
  data_info=DataInfoCfg
)

def main(preprocessor: Preprocessor):
  preprocessor.execute()

if __name__ == "__main__":
  run_hydra(main, Config)
