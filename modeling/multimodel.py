from abc import ABC, abstractmethod

from torch import nn

from lightning import LightningModule

from .mtl_loss import construct_mtl_loss
from .tasks import construct_tasks
  
class BaseMultiModel(ABC, LightningModule):
  def __init__(self, config):
    ABC.__init__(self)
    LightningModule.__init__(self)

    self.config = config
    self.tasks = nn.ModuleDict(construct_tasks(config)).cuda()
    self.mtl_loss = construct_mtl_loss(config, self.tasks)
    self.norm_layers = None
  
  def set_norm_layers(self, value):
    self.norm_layers = value

  @abstractmethod
  def forward_base(self, batch):
    pass
    
  def split_step(self, named_batches, split):
    hiddens = {}
    for set_name in self.config["flat_datasets"]:
      hiddens[set_name] = self.forward_base(named_batches[set_name])
    
    batch_sizes, metricss, losses = {}, {}, {}
    for set_name, task_name in self.config["melt_pairs"]:
      batch = named_batches[set_name]
      task = self.tasks[task_name]
      hidden = hiddens[set_name]

      batch_size = next(iter(batch.values())).shape[0]

      metricss[task_name], losses[task_name] = task.compute(hidden, batch, split)
      batch_sizes[task_name] = batch_size
    
    log_kwargs = {"prog_bar": split == "test", "on_epoch": True, "on_step": False}
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
    losses = [losses[task_name] for task_name in self.config["melt_tasks"]]
    return self.mtl_loss(losses, self.norm_layers, split)
    
  def training_step(self, batch):
    return self.split_step(batch, "train")
 
  def validation_step(self, batch, _):
    return self.split_step(batch, "valid")
  
  def test_step(self, batch, _):
    return self.split_step(batch, "test")

  def on_split_epoch_end(self, split):
    for task in self.tasks.values():
      task.metrics[split].reset()
    self.mtl_loss.reset_norms()

  def on_train_epoch_end(self):
    self.on_split_epoch_end("train")
  
  def on_validation_epoch_end(self):
    self.on_split_epoch_end("valid")
    
  def on_test_epoch_end(self):
    self.on_split_epoch_end("test")

