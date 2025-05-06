import numpy as np
import torch

from adapters import MTLLoRAConfig
import adapters.composition as ac

from .base_model import BaseModel

class MTLLoraModel(BaseModel):
  def __init__(self, config):
    super().__init__(config, do_quantize=False)

    adapter_config = MTLLoRAConfig(
      r=config["adapter_r"],
      dropout=config["adapter_dropout"],
      vera_d=config["adapter_d"],
      vera_b=config["adapter_b"],
    )
    for task in config["melt_tasks"]:
      self.model.add_adapter(task, adapter_config)
    self.model.share_parameters(adapter_names=config["melt_tasks"])
    self.model.set_active_adapters(ac.MultiTask(*config["melt_tasks"]))
    self.model.train_adapter(ac.MultiTask(*config["melt_tasks"]))

  def forward_encoder(self, batch, i):
    return self.model(
      batch["tokens"],
      attention_mask=batch["mask"].float(),
      task_ids=torch.full((batch["size"],), i),
    ).last_hidden_state
  
  def forward(self, batches, split):
    metricss, losses, sizes = {}, {}, {}
    for i, (name, batch) in enumerate(batches.items()):
      hidden = self.forward_encoder(batch, i)
      metricss[name], losses[name] = self.tasks[name].compute(hidden, batch, split)
      sizes[name] = batch["size"]

    loss_counts = torch.Tensor(np.concatenate([
      np.full(self.tasks[name].loss_dim, sizes[name])
      for name in self.config["melt_tasks"]
    ], axis=0)).cuda()
    loss_weights = loss_counts / loss_counts.mean()

    return metricss, losses, sizes, loss_weights
