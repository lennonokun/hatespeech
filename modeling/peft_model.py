
import torch

from peft import get_peft_model, LoraConfig, TaskType

from .base_model import BaseModel

class PeftModel(BaseModel):
  def __init__(self, config):
    super().__init__(config, do_quantize=True)

    # TODO check more parameters
    peft_config = LoraConfig(
      task_type=TaskType.FEATURE_EXTRACTION,
      inference_mode=False,
      r=config["adapter_r"],
      lora_alpha=config["adapter_alpha"],
      lora_dropout=config["adapter_dropout"],
      # target_modules=["query", "key", "value"],
      target_modules=["query", "key"],
      use_rslora=True,
    )
    self.model = get_peft_model(self.model, peft_config) # pyright: ignore
    self.model.print_trainable_parameters()
    # print([(k,v.dtype) for k,v in self.model.named_parameters() if v.requires_grad])

  def forward(self, batches, split):
    tokens = torch.cat([batches[n]["tokens"] for n in self.config["flat_datasets"]], dim=0)
    mask = torch.cat([batches[n]["mask"] for n in self.config["flat_datasets"]], dim=0)
    output = self.model(input_ids=tokens, attention_mask=mask.bfloat16())
    hidden = output.last_hidden_state.float()

    offset = 0
    metricss, losses, sizes = {}, {}, {}
    for set_name in self.config["flat_datasets"]:
      batch = batches[set_name]
      hidden2 = hidden[offset: offset+batch["size"]]
      for task_name in self.config["active_tasks"][set_name]:
        task = self.tasks[task_name]
        metricss[task_name], losses[task_name] = task.compute(hidden2, batch, split)
        sizes[task_name] = batch["size"]
      offset += batch["size"]

    loss_counts = torch.cat([
      torch.full(
        (self.tasks[task_name].loss_dim,),
        batches[set_name]["size"],
        dtype=torch.float32,
        device=self.model.device
      ) for set_name, task_name in self.config["melt_pairs"]
    ], dim=0)
    loss_weights = loss_counts / loss_counts.mean()

    return metricss, losses, sizes, loss_weights
