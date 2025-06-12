from typing import *
# import warnings

from lightning import Trainer
import torch

from hydra_zen import store, zen, make_config
from hydra.conf import HydraConf, JobConf

# from preprocessing import load_stats, do_fix, construct_preprocessor
from modeling import *

# TODO restructure like lightning-hydra-zen-template
# TODO standardize TaskSet vs Heads (set or plural)
# TODO add PreTrainedModel to custom types?
# TODO separate heads loss + metrics ? (loss could go to mtlloss?)
# TODO re-add task-combined dataset for mtllora? (but it performed much more poorly)
# TODO make sure WeightedSampling good
# TODO make HateDatasets something before HateDataModule?
# TODO HateHeads hidden_size 
# TODO top-level + interpolated stores vs packaged stores
# TODO just one paths/info config instead of stats + datasetinfo

TrainCfg = make_config(
  hydra_defaults=[
    "_self_",
    # misc
    {"tasks": ["label"]},
    # trainer
    {"callbacks": "default"},
    {"logger": "mlflow"},
    # module
    {"method": "lora8"},
    {"model": "electra-small"},
    {"quantization": "nf4-double"},
    {"optimization": "default"},
    {"mtl_loss": "rw"},
    {"heads": "default"},
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

def train(module: HateModule, datamodule: HateDatamodule, trainer: Trainer):
  torch.cuda.empty_cache()
  torch.set_float32_matmul_precision("medium")

  trainer.fit(module, datamodule=datamodule)
  trainer.test(module, datamodule=datamodule)

  # module = torch.compile(module, dynamic=False)

  # tuner = Tuner(trainer)
  # tuner.lr_find(module, datamodule=datamodule, min_lr=1e-6, max_lr=2e-3)
  # tuner.scale_batch_size(module, datamodule=data, init_val=32, max_trials=3)
 
  # torch.cuda.empty_cache()
  # trainer.logger.log_hyperparams(cfg) # pyright: ignore[reportOptionalMemberAccess]
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
