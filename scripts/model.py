import torch as pt
from typing import *
from lightning import Trainer

from common import make_config, run_hydra, DataInfoCfg
from modeling import TrainerCfg, HateModuleCfg, HateDatamoduleCfg
from modeling import HateModule, HateDatamodule 

Config = make_config(
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
    {"optimization": "fastest"},
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
  data_info=DataInfoCfg,
  # misc
  action="train",
  save_path=None,
  load_path=None,
  vis_params=False,
)

def main(
  module: HateModule,
  datamodule: HateDatamodule,
  trainer: Trainer,
  action: str,
  vis_params: bool,
):
  pt.cuda.empty_cache()
  pt.set_float32_matmul_precision("medium")

  if vis_params:
    module.vis_params()

  if action == "train":
    trainer.fit(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule)
  elif action == "test":
    trainer.test(module, datamodule=datamodule)
  else:
    raise ValueError(f"invalid action specified: {action}")

  module.save(action)

if __name__ == "__main__":
  run_hydra(main, Config)
