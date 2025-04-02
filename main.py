import os
import warnings
import argparse
from dotenv import load_dotenv

import torch
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from lightning.pytorch.loggers import TensorBoardLogger
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING
from datasets.utils.logging import disable_progress_bar

from data import HateData, HateAugmenter
from module import HateModule
from visualize import HateVisualizer

def do_train(config):
  torch.cuda.empty_cache()
  data = HateData(config)
  module = HateModule(config)
  
  trainer = Trainer(
    enable_checkpointing=True,
    max_epochs=30,
    # limit_train_batches=0.5,
    # limit_val_batches=0.2,
    accelerator="auto",
    # gradient_clip_val=5.0,
    precision="bf16-mixed",
    logger=TensorBoardLogger("tb_logs", name="hatexplain"),
    devices=1,
    callbacks=[
      # EarlyStopping(monitor="valid_label_f1", min_delta=0.01, patience=config["patience"], mode="max", verbose=True),
      EarlyStopping(monitor="valid_loss", min_delta=0.1, patience=config["patience"], verbose=True),
    ],
  )
  # trainer.logger.log_hyperparams(config)
  trainer.fit(module, datamodule=data)
  trainer.test(module, datamodule=data)

def do_test(config):
  data = HateData(config)
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
  data = HateData(config)
  data.setup("")
  
def do_augment(config):
  augmenter = HateAugmenter(config)

def do_visualize(config):
  visualizer = HateVisualizer(config)
  visualizer.visualize_dataset()
  # visualizer.visualize_repl()

if __name__ == "__main__":
  parser = argparse.ArgumentParser("hatespeech")
  parser.add_argument("mode", type=str)
  args = parser.parse_args()

  load_dotenv()
  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  torch.set_float32_matmul_precision("medium")
  warnings.filterwarnings('ignore', message=EPOCH_DEPRECATION_WARNING[:10], category=UserWarning)
  disable_progress_bar()

  config = {
    "model": "google/electra-small-discriminator",
    "best_model": "tb_logs/hatexplain/version_57/checkpoints/epoch=16-step=2142.ckpt",
    "targets": [
      "African", "Arab", "Asian", "Caucasian",
      "Hispanic", "Homosexual", "Islam", "Jewish",
      "Other", "Refugee", "Women"
    # ignored bc frequency threshold 0.005 or None
    ], "targets_ignore": [
      "None", "Asexual", "Bisexual", "Disability",
      "Economic", "Heterosexual", "Hindu", "Indian",
      "Indigenous", "Men", "Minority", "Nonreligious", 
    ],
    "labels": ["hatespeech", "normal"],
    "multitask_targets": True,
    "num_targets": 11,
    "num_labels": 2,
    "num_annotators": 3,
    "num_augments": 3,
    "num_tasks": 3,
    "do_augment": True,
    "augmented_path": "data/augmented.parquet",
    "augment_batch_size": 32,
    "augment_num_workers": 5,
    "logging": "terminal",
    "tokenize_batch_size": 64,
    "max_length": 128,
    "batch_size": 128,
    "learning_rate": 5e-5,
    "patience": 3,
    "env_file": ".env",
  }

  mode_methods = {
    "train": do_train,
    "augment": do_augment,
    "visualize": do_visualize,
    "test": do_test,
    "load": do_load,
  }

  if current_method := mode_methods.get(args.mode):
    current_method(config)
  else:
    print(f"Invalid mode '{args.mode}'")
