import numpy as np
import torch
from torch.nn import functional as F
from torch import nn

from lightning import LightningModule
from torch.optim import AdamW
from .tasks import TargetTask, RationaleTask, LabelTask, ScoreTask

from adapters import AutoAdapterModel, MTLLoRAConfig
import adapters.composition as ac

class HateMTLLora(LightningModule):
  _task_constructors = {
    "target": TargetTask,
    "rationale": RationaleTask,
    "label": LabelTask,
    "score": ScoreTask,
  }

  def __init__(self, config):
    super().__init__()
    self.config = config
    self.model = AutoAdapterModel.from_pretrained(
      config["model"],
      low_cpu_mem_usage=True,
    )

    adapter_config = MTLLoRAConfig(
      r=config["adapter_r"],
      dropout=config["adapter_dropout"],
      n_up_projection=3,
    )
    for task in config["melt_tasks"]:
      self.model.add_adapter(task, adapter_config)
    self.model.share_parameters(adapter_names=config["melt_tasks"])
    self.model.set_active_adapters(ac.MultiTask(*config["melt_tasks"]))
    self.model.train_adapter(ac.MultiTask(*config["melt_tasks"]))

    self.tasks = nn.ModuleDict({
      name: self._task_constructors[name](config)
      for name in config["melt_tasks"]
    })
    self.tasks = self.tasks.cuda()

  def forward(self, batches):
    features, labels, task_ids = batches

    return self.model(
      features["tokens"],
      attention_mask=features["mask"].float(),
      task_ids=task_ids
    ).last_hidden_state

  # TODO unify features but split tasks?
  def split_step(self, batches, split):
    features, labels, task_ids = batches

    hidden = self.forward(batches)

    batch_sizes, metricss, losses = {}, {}, {}
    for i, (name, batch_label) in enumerate(labels.items()):
      hidden_task = hidden[task_ids == i]
      batch_size = batch_label.shape[0]

      # TODO doesnt work for rationales, maybe just flatten fully idk
      batch_label = {name: batch_label}
      for feature, values in features.items():
        batch_label[feature] = values[task_ids == i]

      metricss[name], losses[name] = self.tasks[name].compute(hidden_task, batch_label, split)
      batch_sizes[name] = batch_size
    
    log_kwargs = {"prog_bar": split == "test", "on_epoch": True, "on_step": True}

    # log metrics
    for name, metrics in metricss.items():
      self.log_dict(metrics, batch_size=batch_sizes[name], **log_kwargs)

    # log losses
    if split != "test":
      for name, loss in losses.items():
        b_size = batch_sizes[name]
        if loss.dim() == 0:
          self.log(f"{split}_{name}_loss", loss, batch_size=b_size, **log_kwargs)
        elif loss.dim() == 1:
          for i, sub_loss in enumerate(loss):
            self.log(f"{split}_{name}_loss_{i}", sub_loss, batch_size=b_size, **log_kwargs)

    # ensure that order is consistent
    # losses_list = [v for _,v in sorted(losses.items())]
    return sum(torch.mean(loss) for loss in losses.values())

  def configure_optimizers(self):
    std_params = filter(lambda p: p.requires_grad, self.parameters())
    return AdamW(params=std_params, lr=self.config["learning_rate"])
    
  def training_step(self, batch):
    return self.split_step(batch, "train")
 
  def validation_step(self, batch, _):
    return self.split_step(batch, "valid")
  
  def test_step(self, batch, _):
    return self.split_step(batch, "test")

  def on_split_epoch_end(self, split):
    for task in self.tasks.values():
      task.metrics[split].reset()

  def on_train_epoch_end(self):
    self.on_split_epoch_end("train")
  
  def on_validation_epoch_end(self):
    self.on_split_epoch_end("valid")
    
  def on_test_epoch_end(self):
    self.on_split_epoch_end("test")

  # def on_after_backward(self):
  #   for name, param in self.named_parameters():
  #     if param.requires_grad and param.grad is not None:
  #       print(f"{name} gradient: {param.grad.norm()}")


