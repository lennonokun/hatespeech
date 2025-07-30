from preprocessing import fix_explain
from common import DataInfoCfg, run_hydra, make_config

Config = make_config(data_info=DataInfoCfg)

if __name__ == "__main__":
  run_hydra(fix_explain, Config)
