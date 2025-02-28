import os
import warnings
import argparse

import torch
from lightning import Trainer
from lightning.pytorch.callbacks.early_stopping import EarlyStopping
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING
from datasets.utils.logging import disable_progress_bar

from data import HateData, HateAugmenter
from module import HateModule

def do_train(config):
  torch.cuda.empty_cache()
  data = HateData(config)
  module = HateModule(config)
  
  trainer = Trainer(
    enable_checkpointing=True,
    max_epochs=100,
    # limit_train_batches=0.5,
    # limit_val_batches=0.2,
    accelerator="auto",
    precision="bf16-mixed",
    gradient_clip_val=10.0,
    devices=1,
    callbacks=[
      EarlyStopping(monitor="valid_f1", min_delta=0.01, patience=3, mode="max", verbose=True),
    ],
  )
  trainer.logger.log_hyperparams(config)
  trainer.fit(module, datamodule=data)
  trainer.test(module, datamodule=data)

def do_augment(config):
  augmenter = HateAugmenter(config)
  augmenter.augment()
  augmenter.save()

if __name__ == "__main__":
  parser = argparse.ArgumentParser("hatespeech")
  parser.add_argument("mode", type=str)
  args = parser.parse_args()

  os.environ["TOKENIZERS_PARALLELISM"] = "false"
  torch.set_float32_matmul_precision("medium")
  warnings.filterwarnings('ignore', message=EPOCH_DEPRECATION_WARNING[:10], category=UserWarning)
  disable_progress_bar()

  config = {
    "model": "google/electra-small-discriminator",
    "num_labels": 3,
    "num_augments": 2,
    "augmented_path": "data/augmented.hf",
    "max_length": 128,
    "batch_size": 128,
    "learning_rate": 1e-5,
  }

  if args.mode == "train":
    do_train(config)
  elif args.mode == "augment":
    do_augment(config)
