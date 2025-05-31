# import warnings

# from preprocessing import load_stats, do_fix, construct_preprocessor
from modeling import *

from hydra_zen import store, zen
from hydra.conf import HydraConf, JobConf

import torch

# TODO standardize TaskSet vs Heads (set or plural)
# TODO add PreTrainedModel to custom types?
# TODO separate heads loss + metrics ? (loss could go to mtlloss?)
# TODO re-add task-combined dataset for mtllora? (but it performed much more poorly)
# TODO make sure WeightedSampling good
# TODO make HateDatasets something before HateDataModule?
# TODO fix cwd?
# TODO why datamodule required to be a different var but not heads or mtl_loss
@store(
  name="train",
  encoder=EncoderCfg,
  optimization=OptimizationCfg,
  tasks=TaskSetCfg,
  stats=StatsCfg,
  heads=HateHeadsCfg,
  mtl_loss=MTLLossCfg,
  trainer=TrainerCfg,
  datamodule=HateDatamoduleCfg,
)
def train(
  encoder: AdapterEncoder,
  optimization: HateOptimization,
  tasks: TaskSet,
  stats: Stats,
  heads: PartialHeads,
  mtl_loss: PartialMTLLoss,
  datamodule: PartialDatamodule,
  trainer: Trainer,
):
  heads = heads(tasks=tasks, encoder=encoder, stats=stats)
  mtl_loss = mtl_loss(tasks=tasks)

  module = HateModule(encoder, heads, optimization, tasks, mtl_loss)
  datamodule2 = datamodule(tasks=tasks) # TODO whyy

  torch.cuda.empty_cache()
  torch.set_float32_matmul_precision("medium")

  trainer.fit(module, datamodule=datamodule2)
  trainer.test(module, datamodule=datamodule2)

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
  store.add_to_hydra_store()
  zen(train).hydra_main(
    config_name="train",
    version_base="1.1",
    config_path=".",
  )
