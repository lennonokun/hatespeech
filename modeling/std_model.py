import numpy as np
import torch

import adapters
from adapters import LoRAConfig

from .base_model import BaseModel

class StandardModel(BaseModel):
  def __init__(self, config):
    super().__init__(config, do_quantize=True)

    adapters.init(self.model)
    lora_config = LoRAConfig(
      r=config["adapter_r"],
      alpha=config["adapter_alpha"],
      dropout=config["adapter_dropout"],
      attn_matrices=["q","k","v"],
      selfattn_lora=True,
      # intermediate_lora=True,
      # output_lora=True,
    )
    self.model.add_adapter("qlora", lora_config)
    self.model.set_active_adapters("qlora")
    self.model.train_adapter("qlora")

    # self.norm_layers = [
    #   v for k,v in self.model.named_parameters()
    #   if ("layer.8" in k) and "lora" in k
    # ]

    # print([(k,v.dtype) for k,v in self.model.named_parameters()])
    for name, param in self.model.named_parameters():
      if "lora" in name:
        param.data = param.data.to(torch.bfloat16)

  def forward(self, batches, split):
    metricss, losses, sizes = {}, {}, {}
    for set_name in self.config["flat_datasets"]:
      batch = batches[set_name]
      hidden = self.model(
        input_ids=batch["tokens"],
        attention_mask=batch["mask"].bfloat16()
      ).last_hidden_state.float()

      for task_name in self.config["active_tasks"][set_name]:
        task = self.tasks[task_name]
        metricss[task_name], losses[task_name] = task.compute(hidden, batch, split)
        sizes[task_name] = batch["size"]

    loss_counts = torch.Tensor(np.concatenate([
      np.full(self.tasks[task_name].loss_dim, batches[set_name]["size"])
      for set_name, task_name in self.config["melt_pairs"]
    ], axis=0)).to(self.model.device)
    loss_weights = loss_counts / loss_counts.mean()

    return metricss, losses, sizes, loss_weights
