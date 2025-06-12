from typing import *

from lightning import Trainer
from lightning.pytorch.callbacks import Callback, RichProgressBar
from lightning.pytorch.loggers import TensorBoardLogger, MLFlowLogger
# from lightning.pytorch.tuner.tuning import Tuner

from omegaconf import MISSING
from hydra_zen import store, make_config

from .utils import *
from .tasks import TaskSet

class MultiEarlyStopping(Callback):
  def __init__(
      self,
      tasks: TaskSet,
      patience: float,
      num_required: int,
      wait_initial: int
  ):
    super().__init__()
    self.monitors = tasks.monitors()
    self.patience = patience
    self.num_required = num_required

    self.wait_initial = wait_initial
    self.wait_count = 0
    self.prev_scores = {name: float("-inf") * thresh for name, thresh in self.monitors.items()}

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
        # TODO assert instead bc not necessary functionality?
        metric = trainer.callback_metrics.get(name)
        if metric is not None:
          self.prev_scores[name] = metric.squeeze()
    else:
      self.wait_count += 1
      if self.wait_count >= self.patience:
        trainer.should_stop = True

  def on_validation_end(self, trainer, pl_module):
    self._check_early_stopping(trainer)

def make_callbacks(**kwargs: Callback) -> List[Callback]:
  return list(kwargs.values())

RichProgressBarCfg = fbuilds(
  RichProgressBar,
  leave=True
)
MultiEarlyStoppingCfg = fbuilds(
  MultiEarlyStopping,
  tasks="${tasks}",
  num_required=1,
  patience=3,
  wait_initial=4,
)

callbacks_store = store(group="callbacks", to_config=remove_types)
callbacks_store(fbuilds(make_callbacks), name="none")
callbacks_store(fbuilds(
  make_callbacks,
  progress_bar=RichProgressBarCfg,
  early_stopping=MultiEarlyStoppingCfg
), name="default")

logger_store = store(group="logger", to_config=remove_types)
logger_store(fbuilds(
  MLFlowLogger,
  experiment_name="hatespeech",
  tracking_uri="file:./mlruns",
), name="mlflow")
logger_store(fbuilds(
  TensorBoardLogger,
  save_dir=MISSING,
  name="hatespeech",
), name="tensorboard")

TrainerCfg = fbuilds(
  Trainer,
  callbacks="${callbacks}",
  logger="${logger}",
  max_epochs=100,
  enable_checkpointing=False,
  accelerator="auto",
  gradient_clip_val=0.5,
  devices=1,
  # limit_train_batches=1.0,
)
