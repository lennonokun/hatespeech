import warnings
import argparse
import subprocess

import torch
from lightning import Trainer
from lightning.pytorch.callbacks import RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger
from torch.optim.lr_scheduler import EPOCH_DEPRECATION_WARNING

from preprocessing import load_stats, do_fix, ExplainPreprocessor, MeasuringPreprocessor
from modeling import HateModule, HateDatamodule, HateVisualizer, MultiEarlyStopping
# from modeling import HateMTLLora

def do_train(config):
  torch.cuda.empty_cache()
  # if config["mtl_lora"]:
  #   data = HateDatamodule(config, "task")
  #   module = HateMTLLora(config)
  # else:
  #   data = HateDatamodule(config, "dataset")
  #   module = HateModule(config)
  data = HateDatamodule(config)
  module = HateModule(config)
 
  trainer_kwargs = {
    "enable_checkpointing": True,
    "max_epochs": 20,
    "accelerator": "auto",
    # "gradient_clip_val": 5.0,
    "precision": "bf16-mixed",
    "devices": 1,
    "callbacks": [
      MultiEarlyStopping(config["stopping_monitors"], patience=2, num_required=2),
      RichProgressBar(leave=True)
    ]
  }
  if config["quick_model"]:
    trainer_kwargs["limit_train_batches"] = 0.2
    trainer_kwargs["limit_val_batches"] = 0.2
  else:
    trainer_kwargs["logger"] = MLFlowLogger("hatespeech", tracking_uri="file:./mlruns")

  trainer = Trainer(**trainer_kwargs)
  trainer.logger.log_hyperparams(config) # pyright: ignore[reportOptionalMemberAccess]
  trainer.fit(module, datamodule=data)
  trainer.test(module, datamodule=data)

def do_test(config):
  data = HateDatamodule(config, "dataset")
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
    "num_hidden": 256,
    "max_length": 128,
    "batch_size": 142,
    "patience": 4,
    "learning_rate": 5e-5,
    "adapter_r": 16,
    "adapter_dropout": 0.05,
    "quick_model": False,
    # task + dataset selection
    "features": ["tokens", "mask"],
    "active_tasks": {
      "explain": ["target", "rationale", "label"],
      # "measuring": ["score"],
      # "explain": ["target", "label"],
    },
    "stopping_monitors": {
      "valid_target_f1": 0.02,
      # "valid_rationale_f1": 0.01,
      "valid_label_f1": 0.01,
      # "valid_score_mse": -0.01,
    },
    # MTL + VAT/GAT
    "mtl_lora": False,
    "mtl_importances": {
      "label": 1e0,
      "target": 1e0,
      "rationale": 1e0,
      "score": 2e0,
    },
    "mtl_expand_targets": True,
    "mtl_norm_do": True,
    "mtl_norm_period": 4,
    "mtl_norm_length": 8,
    "mtl_weighing": "dwa",
    "mtl_dwa_T": 2.0,
    "vat_epsilon": 0.0,
  }
  # more data
  config["melt_pairs"] = [
    (dataset, task)
    for dataset, tasks in config["active_tasks"].items()
    for task in tasks
  ]
  config["melt_tasks"] = [task for _, task in config["melt_pairs"]]
  config["melt_datasets"] = [dataset for dataset, _ in config["melt_pairs"]]
  config["flat_datasets"] = [dataset for dataset in config["active_tasks"].keys()]
  # config["flat_tasks"] = [task for tasks in config["active_datasets"] for task in tasks]
  config["num_target"] = len(config["cats_target"])
  config["num_label"] = len(config["cats_label"])
  config["cols_target"] = [f"target_{cat}" for cat in config["cats_target"]]
  config["cols_label"] = [f"label_{cat}" for cat in config["cats_label"]]
  config["stats"] = load_stats(config)
  config["git_commit"] = subprocess.check_output(
    ['git', 'rev-parse', '--short', 'HEAD']
  ).decode('ascii').strip()

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
