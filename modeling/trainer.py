from typing import * # pyright: ignore[reportWildcardImportFromLibrary]
from hydra_zen import make_custom_builds_fn

from lightning import Trainer
from lightning.pytorch.tuner.tuning import Tuner
from lightning.pytorch.callbacks import Callback, RichProgressBar
from lightning.pytorch.loggers import MLFlowLogger

class MultiEarlyStopping(Callback):
  def __init__(self, monitors: Dict[str, float], patience: float, num_required: int, wait_initial: int):
    super().__init__()
    self.monitors = monitors
    self.patience = patience
    self.num_required = num_required

    self.wait_initial = wait_initial
    self.wait_count = 0
    self.prev_scores = {
      name: float("-inf") * thresh for name, thresh in monitors.items()
    }

  def _check_early_stopping(self, trainer):
    if self.wait_initial > 0:
      self.wait_initial -= 1
      return
    
    num_improved = 0
    for name, thresh in self.monitors.items():
      if name in trainer.callback_metrics:
        curr_score = trainer.callback_metrics[name].squeeze()
        is_improved = ((curr_score - self.prev_scores[name]) / thresh) >= 1.0
        num_improved += bool(is_improved)

    if num_improved >= self.num_required:
      self.wait_count = 0
      for name in self.monitors:
        metric = trainer.callback_metrics.get(name)
        if metric is not None:
          self.prev_scores[name] = metric.squeeze()
    else:
      self.wait_count += 1
      if self.wait_count >= self.patience:
        trainer.should_stop = True

  def on_validation_end(self, trainer, pl_module):
    self._check_early_stopping(trainer)

# TODO allow for selection?
def build_callbacks(progress: Callback, stopping: Callback):
  return [progress, stopping]

builds = make_custom_builds_fn(populate_full_signature=True)

MultiEarlyStoppingCfg = builds(
  MultiEarlyStopping,
  monitors = {
    "valid_label_f1": 5e-3,
    "valid_target_f1": 5e-3,
    "valid_rationale_f1": 1e-2,
    "valid_score_mse": -2e-2,
  },
  num_required = 1,
  patience = 3,
  wait_initial = 4,
)

RichProgressBarCfg = builds(
  RichProgressBar,
  leave=True
)

CallbacksCfg = builds(
  build_callbacks,
  progress=RichProgressBarCfg,
  stopping=MultiEarlyStoppingCfg,
)

LoggerCfg = builds(
  MLFlowLogger,
  experiment_name = "hatespeech",
  tracking_uri = "file:./mlruns",
)

TrainerCfg = builds(
  Trainer,
  callbacks = CallbacksCfg,
  logger = LoggerCfg,
  max_epochs = 100,
  enable_checkpointing = False,
  accelerator = "auto",
  gradient_clip_val = 0.5,
  devices = 1,
  limit_train_batches = 1.0,
)
