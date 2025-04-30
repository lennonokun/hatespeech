import numpy as np
import torch
from torch.optim import AdamW

from adapters import AutoAdapterModel, LoRAConfig

from .base_model import BaseModel

class StandardModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)

    self.model = AutoAdapterModel.from_pretrained(
      config["model"],
      low_cpu_mem_usage=True,
    )
    self.model.add_adapter("my_adapter", config=LoRAConfig(
      r=config["adapter_r"],
      dropout=config["adapter_dropout"],
    ))
    self.model.set_active_adapters("my_adapter")
    self.model.train_adapter("my_adapter")
  
    self.norm_layers = [
      v for k,v in self.model.named_parameters()
      if ("layer.10" in k or "layer.11" in k) and "lora" in k
    ]

  def forward(self, batches, split):
    hiddens = {}
    for set_name in self.config["flat_datasets"]:
      batch = batches[set_name]
      outputs = self.model(batch["tokens"], attention_mask=batch["mask"].float())
      hiddens[set_name] = outputs.last_hidden_state
    
    metricss, losses, sizes = {}, {}, {}
    for set_name, task_name in self.config["melt_pairs"]:
      batch = batches[set_name]
      task = self.tasks[task_name]
      hidden = hiddens[set_name]

      metricss[task_name], losses[task_name] = task.compute(hidden, batch, split)
      sizes[task_name] = batch["size"]

    loss_counts = torch.Tensor(np.concatenate([
      np.full(self.tasks[task_name].loss_dim, batches[set_name]["size"])
      for set_name, task_name in self.config["melt_pairs"]
    ], axis=0)).cuda()
    loss_weights = loss_counts / loss_counts.mean()

    return metricss, losses, sizes, loss_weights

  def configure_optimizers(self):
    std_params = filter(lambda p: p.requires_grad, self.parameters())
    return AdamW(params= std_params, lr=self.config["learning_rate"])
