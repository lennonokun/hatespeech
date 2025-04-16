import os
import warnings
import argparse

import torch
from lightning import Trainer
# from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING

from preprocessing import do_fix, ExplainPreprocessor, MeasuringPreprocessor
from modeling import HateDatamodule, HateModule, HateVisualizer

def do_train(config):
  torch.cuda.empty_cache()
  data = HateDatamodule(config)
  module = HateModule(config)
  
  trainer_kwargs = {
    "enable_checkpointing": True,
    "max_epochs": 30,
    "accelerator": "auto",
    # "gradient_clip_val": 5.0,
    "precision": "bf16-mixed",
    "logger": TensorBoardLogger("tb_logs", name="hatexplain"),
    "devices": 1,
    "callbacks": [
      # EarlyStopping(monitor="valid_loss", min_delta=0.05, patience=config["patience"], verbose=True),
    ],
  }
  if config["quick_model"]:
    trainer_kwargs["limit_train_batches"] = 0.2
    trainer_kwargs["limit_val_batches"] = 0.2

  trainer = Trainer(**trainer_kwargs)
  trainer.fit(module, datamodule=data)
  trainer.test(module, datamodule=data)

def do_test(config):
  data = HateDatamodule(config)
  module = HateModule.load_from_checkpoint(
    config["best_model"],
    map_location=torch.device("cuda"),
  )
  trainer = Trainer(
    accelerator="auto",
    precision="bf16-mixed",
    devices=1,
  )

  trainer.test(module, datamodule=data)

def do_load(config):
  data = HateDatamodule(config)
  data.setup("")

def do_preprocess(config):
  preprocessor = ExplainPreprocessor(config)
  preprocessor.execute()

def do_preprocess2(config):
  preprocessor = MeasuringPreprocessor(config)
  preprocessor.execute()
  
def do_visualize(config):
  visualizer = HateVisualizer(config)
  visualizer.visualize_dataset()
  # visualizer.visualize_repl()

if __name__ == "__main__":
  parser = argparse.ArgumentParser("hatespeech")
  parser.add_argument("mode", type=str)
  args = parser.parse_args()

  # os.environ["TOKENIZERS_PARALLELISM"] = "false"
  torch.set_float32_matmul_precision("medium")
  warnings.filterwarnings('ignore', message=EPOCH_DEPRECATION_WARNING[:10], category=UserWarning)

  config = {
    # data misc
    "cats_target": [
      "African", "Arab", "Asian", "Caucasian", "Hispanic",
      "Homosexual", "Islam", "Jewish", "Other", "Refugee", "Women"
    ],
    "cats_label": ["hatespeech", "offensive", "normal"],
    "round_train": ["target", "label", "rationale"],
    "tokenize_batch_size": 64,
    # preprocessing paths
    "explain_dirty_path": "data/explain/dirty.json",
    "input_dataset_path": "data/{name}/input.parquet",
    "output_dataset_path": "data/{name}/output_{split}.parquet",
    "output_stats_path": "data/{name}/stats.json",
    # modeling misc
    "model": "google/electra-small-discriminator",
    "best_model": "tb_logs/hatexplain/version_57/checkpoints/epoch=16-step=2142.ckpt",
    "max_length": 128,
    "batch_size": 142,
    "patience": 3,
    "learning_rate": 5e-5,
    "logging": "terminal",
    "quick_model": False,
    # MTL + VAT/GAT
    "num_tasks": 3,
    "multitask_targets": True,
    "mtl_norm_initial": True,
    "mtl_norm_length": 10,
    "mtl_weighing": "dwa",
    "mtl_dwa_T": 2.0,
    "vat_epsilon": 0.0,
  }
  # more data
  config["num_target"] = len(config["cats_target"])
  config["num_label"] = len(config["cats_label"])
  config["cols_target"] = [f"target_{cat}" for cat in config["cats_target"]]
  config["cols_label"] = [f"label_{cat}" for cat in config["cats_label"]]
  
  mode_methods = {
    "fix": do_fix,
    "preprocess": do_preprocess,
    "preprocess2": do_preprocess2,
    "train": do_train,
    "load": do_load,
    "visualize": do_visualize,
  }

  if current_method := mode_methods.get(args.mode):
    current_method(config)
  else:
    print(f"Invalid mode '{args.mode}'")
