
from torch.optim import AdamW

import adapters
from adapters import DiReftConfig
from transformers import AutoModel 

from .multimodel import BaseMultiModel
from .tasks import TargetTask, RationaleTask, LabelTask

class HateModule(BaseMultiModel):
  def __init__(self, config):

    tasks = {
      "target": TargetTask(config),
      "rationale": RationaleTask(config),
      "label": LabelTask(config),
    }
    super().__init__(config, tasks)

    self.model = AutoModel.from_pretrained(
      config["model"],
      low_cpu_mem_usage=True,
    )
    adapters.init(self.model)
    self.model.add_adapter("adapter", DiReftConfig(r=8))
    self.model.set_active_adapters("adapter")

  def forward_base(self, batch):
    outputs = self.model(batch["tokens"], attention_mask=batch["mask"].float())
    return outputs.last_hidden_state

  def configure_optimizers(self):
    std_params = filter(lambda p: p.requires_grad, self.parameters())
    return AdamW(params= std_params, lr=self.config["learning_rate"])
