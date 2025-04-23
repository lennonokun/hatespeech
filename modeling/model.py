
from torch.optim import AdamW

import adapters
from adapters import DiReftConfig
from transformers import AutoModel 

from .multimodel import BaseMultiModel
from .tasks import TargetTask, RationaleTask, LabelTask, ScoreTask

class HateModule(BaseMultiModel):
  _task_constructors = {
    "explain": {
      "target": TargetTask,
      "rationale": RationaleTask,
      "label": LabelTask,
    }, "measuring": {
      "score": ScoreTask,
    }
  }

  def _select_tasks(self, config):
    task_sets = {}
    for active_set_name, active_tasks in config["active_tasks"].items():
      task_sets[active_set_name] = {
        task_name: self._task_constructors[active_set_name][task_name](config)
        for task_name in active_tasks
      }
    return task_sets
      
  def __init__(self, config):
    super().__init__(config, self._select_tasks(config))

    self.model = AutoModel.from_pretrained(
      config["model"],
      low_cpu_mem_usage=True,
    )
    adapters.init(self.model)
    self.model.add_adapter("adapter", DiReftConfig(
      r=config["adapter_r"],
      dropout=config["adapter_dropout"],
    ))
    self.model.set_active_adapters("adapter")

    norm_layers = [
      v for k,v in self.model.named_parameters()
      if "11.reft_layer" in k or "10.reft_layer" in k
    ]
    self.set_norm_layers(norm_layers)

  def forward_base(self, batch):
    outputs = self.model(batch["tokens"], attention_mask=batch["mask"].float())
    return outputs.last_hidden_state

  def configure_optimizers(self):
    std_params = filter(lambda p: p.requires_grad, self.parameters())
    return AdamW(params= std_params, lr=self.config["learning_rate"])
