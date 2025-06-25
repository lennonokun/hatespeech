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
    {"method": "lora8"},
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
)

def train(
    module: HateModule,
    datamodule: HateDatamodule,
    trainer: Trainer,
):
  torch.cuda.empty_cache()
  torch.set_float32_matmul_precision("medium")

  module.vis_params()
  trainer.fit(module, datamodule=datamodule)
  trainer.test(module, datamodule=datamodule)
  module.save()

  # module = torch.compile(module, dynamic=False)

  # tuner = Tuner(trainer)
  # tuner.lr_find(module, datamodule=datamodule, min_lr=1e-6, max_lr=2e-3)
  # tuner.scale_batch_size(module, datamodule=data, init_val=32, max_trials=3)
 
  # torch.cuda.empty_cache()
  # trainer.fit(module, datamodule=datamodule)
  # # module.dequantize()
  # trainer.test(module, datamodule=datamodule)
  # module.save(args.save_adapters)

if __name__ == "__main__":
  store(HydraConf(job=JobConf(chdir=False)))
  store(TrainCfg, name="config")
  store.add_to_hydra_store(overwrite_ok=True)
  zen(train).hydra_main(
    config_name="config",
    version_base="1.3",
    config_path=None,
  )
