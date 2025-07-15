from typing import *
# import warnings

from lightning import Trainer
import torch

from hydra_zen import store, zen, make_config
from hydra.conf import HydraConf, JobConf

# from preprocessing import load_stats, do_fix, construct_preprocessor
from modeling import *

TrainCfg = make_config(
  hydra_defaults=[
    "_self_",
    # misc
    {"tasks": ["none"]},
    # trainer
    {"callbacks": "default"},
    {"logger": "mlflow"},
    # module
    {"method": "ah_lora16"},
    {"model": "electra-small"},
    {"quantization": "nf4-double"},
    {"optimization": "medium"},
    {"mtl_loss": "rw"},
    {"heads": "medium"},
  ],
  # misc
  tasks=None,
  # trainer
  trainer=TrainerCfg,
  callbacks=None,
  logger=None,
  # module
  module=HateModuleCfg,
  method=None,
  model=None,
  quantization=None,
  optimization=None,
  mtl_loss=None,
  heads=None,
  # datamodule
  datamodule=HateDatamoduleCfg,
  # debug
  debug=HateDebugCfg,
  save_path=None,
  load_path=None,
  action="train",
)

def main(
  module: HateModule,
  datamodule: HateDatamodule,
  debug: HateDebug,
  trainer: Trainer,
  action: str,
):
  torch.cuda.empty_cache()
  torch.set_float32_matmul_precision("medium")

  if debug.vis_params:
    module.vis_params()

  if action == "train":
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)
  elif action == "test":
    trainer.test(module, datamodule=datamodule)
  else:
    raise ValueError(f"invalid action specified: {action}")
  module.save()

if __name__ == "__main__":
  store(HydraConf(job=JobConf(chdir=False)))
  store(TrainCfg, name="config")
  store.add_to_hydra_store(overwrite_ok=True)
  zen(main).hydra_main(
    config_name="config",
    version_base="1.3",
    config_path=None,
  )
