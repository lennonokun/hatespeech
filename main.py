import os
import warnings
import argparse

import torch
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING

from preprocessing import do_fix, HatePreprocessor
from modeling import HateDatamodule, HateModule, HateVisualizer

def do_train(config):
  torch.cuda.empty_cache()
  data = HateDatamodule(config)
  module = HateModule(config)
  
  trainer = Trainer(
    enable_checkpointing=True,
    max_epochs=30,
    # limit_train_batches=0.2,
    # limit_val_batches=0.2,
    accelerator="auto",
    # gradient_clip_val=5.0,
    precision="bf16-mixed",
    logger=TensorBoardLogger("tb_logs", name="hatexplain"),
    devices=1,
    callbacks=[
      # EarlyStopping(monitor="valid_loss", min_delta=0.05, patience=config["patience"], verbose=True),
    ],
  )
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
  preprocessor = HatePreprocessor(config)
  preprocessor.preprocess()
  # preprocessor.load_prev_test()
  preprocessor.write()
  
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
    "model": "google/electra-small-discriminator",
    # "model": "vinai/bertweet-base",
    "best_model": "tb_logs/hatexplain/version_57/checkpoints/epoch=16-step=2142.ckpt",
    "targets": [
      "African", "Arab", "Asian", "Caucasian",
      "Hispanic", "Homosexual", "Islam", "Jewish",
      "Other", "Refugee", "Women"
    ],
    "labels": ["hatespeech", "offensive", "normal"],
    "multitask_targets": True,
    "num_targets": 11,
    "num_labels": 3,
    "num_annotators": 3,
    "round_train": ["target", "label", "rationale"],
    "num_tasks": 3,
    # "num_augments": 3,
    # "do_augment": False,
    # "augmented_path": "data/augmented.parquet",
    "dirty_dataset_path": "data/dataset.json",
    "clean_dataset_path": "data/dataset_clean.json",
    "preprocessed_dataset_paths": {
      "train": "data/dataset_train.parquet",
      "valid": "data/dataset_valid.parquet",
      "test": "data/dataset_test.parquet",
    },
    "stats_path": "data/stats.json",
    "logging": "terminal",
    # "augment_batch_size": 32,
    # "augment_num_workers": 5,
    "tokenize_batch_size": 64,
    "max_length": 128,
    "batch_size": 142,
    "learning_rate": 5e-5,
    "vat_epsilon": 0.6,
    "mtl_norm_initial": True,
    "mtl_norm_length": 10,
    "mtl_weighing": "dwa",
    "mtl_dwa_T": 2.0,
    "patience": 3,
  }

  mode_methods = {
    "fix": do_fix,
    "preprocess": do_preprocess,
    "train": do_train,
    "load": do_load,
    "visualize": do_visualize,
  }

  if current_method := mode_methods.get(args.mode):
    current_method(config)
  else:
    print(f"Invalid mode '{args.mode}'")
