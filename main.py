import warnings
import argparse
import subprocess
import pyjson5

import torch
from lightning import Trainer
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING

from preprocessing import load_stats, do_fix, construct_preprocessor
from modeling import construct_datamodule, construct_module ,MultiEarlyStopping

def get_trainer(config):
  early_stopping = MultiEarlyStopping(
    config["stopping_monitors"],
    patience=config["patience"],
    num_required=config["num_required"]
  )
  trainer_kwargs = {
    "enable_checkpointing": False,
    "max_epochs": 30,
    "accelerator": "auto",
    "gradient_clip_val": 0.5,
    # "precision": "bf16-mixed",
    "devices": 1,
    "callbacks": [early_stopping, RichProgressBar(leave=True)],
    "logger": MLFlowLogger("hatespeech", tracking_uri="file:./mlruns")
  }
  if config["quick_model"]:
    trainer_kwargs["limit_train_batches"] = 0.05
    trainer_kwargs["limit_val_batches"] = 0.05

  return Trainer(**trainer_kwargs)

def do_train(args):
  config = args.config

  datamodule = construct_datamodule(config)
  module = construct_module(config)
  # module = torch.compile(module, dynamic=False)
  trainer = get_trainer(config)

  torch.cuda.empty_cache()
  # tuner = Tuner(trainer)
  # tuner.lr_find(module, datamodule=datamodule, min_lr=2e-5, max_lr=1e-3)
  # tuner.scale_batch_size(module, datamodule=data, init_val=32, max_trials=3)
 
  torch.cuda.empty_cache()
  trainer.logger.log_hyperparams(config) # pyright: ignore[reportOptionalMemberAccess]
  trainer.fit(module, datamodule=datamodule)
  trainer.test(module, datamodule=datamodule)

def do_test(args):
  pass
  # config = args.config
  
  # if config["mtllora"]:
  #   data = HateDatamodule(config, "task")
  #   module_class = MTLLoraModel
  # else:
  #   data = HateDatamodule(config, "dataset")
  #   module_class = StandardModel
  # module = module_class.load_from_checkpoint(
  #   config["best_model"],
  #   map_location=torch.device("cuda"),
  # )

  # trainer = Trainer(
  #   accelerator="auto",
  #   precision="bf16-mixed",
  #   devices=1,
  # )

  # trainer.test(module, datamodule=data)

def do_load(args):
  datamodule = construct_datamodule(args.config, args.method)
  datamodule.setup("")

def do_preprocess(args):
  preprocessor = construct_preprocessor(args.config, args.name)
  preprocessor.execute()

def load_config():
  config = pyjson5.decode_io(open("config.json", "r")) # pyright: ignore
  config["melt_pairs"] = [
    (dataset, task)
    for dataset, tasks in config["active_tasks"].items()
    for task in tasks
  ]
  config["melt_tasks"] = [task for _, task in config["melt_pairs"]]
  config["melt_datasets"] = [dataset for dataset, _ in config["melt_pairs"]]
  config["flat_datasets"] = [dataset for dataset in config["active_tasks"].keys()]
  config["num_target"] = len(config["cats_target"])
  config["num_label"] = len(config["cats_label"])
  config["cols_target"] = [f"target_{cat}" for cat in config["cats_target"]]
  config["cols_label"] = [f"label_{cat}" for cat in config["cats_label"]]
  config["stats"] = load_stats(config)
  config["git_commit"] = subprocess.check_output(
    ['git', 'rev-parse', '--short', 'HEAD']
  ).decode('ascii').strip()
  return config

if __name__ == "__main__":
  # os.environ["TOKENIZERS_PARALLELISM"] = "false"
  torch.set_float32_matmul_precision("medium")
  warnings.filterwarnings('ignore', message=EPOCH_DEPRECATION_WARNING[:10], category=UserWarning)

  parser = argparse.ArgumentParser("hatespeech")
  parser.add_argument("--config", default=load_config())
  subparsers = parser.add_subparsers()

  parser_fix = subparsers.add_parser("fix")
  parser_fix.set_defaults(func=do_fix)

  parser_train = subparsers.add_parser("train")
  parser_train.set_defaults(func=do_train)

  parser_load = subparsers.add_parser("load")
  parser_load.add_argument("method", type=str)
  parser_load.set_defaults(func=do_load)

  parser_preprocess = subparsers.add_parser("preprocess")
  parser_preprocess.add_argument("dataset", type=str)
  parser_preprocess.set_defaults(func=do_preprocess)

  args = parser.parse_args()
  args.func(args)
