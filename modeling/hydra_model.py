import numpy as np
import torch
# from torch.optim import AdamW
from bitsandbytes.optim import *
from torch.optim.lr_scheduler import *

from transformers import BitsAndBytesConfig
from adapters import AutoAdapterModel, LoRAConfig
from adapters import composition as ac

from .base_model import BaseModel

class HydraModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)

    self.model = AutoAdapterModel.from_pretrained(
      config["model"],
      low_cpu_mem_usage=True,
      device_map="auto",
      torch_dtype=torch.bfloat16,
      # quantization_config=BitsAndBytesConfig(
      #   load_in_4bit=True,
      #   bnb_4bit_quant_type="nf4",
      #   bnb_4bit_use_double_quant=True,
      #   bnb_4bit_compute_dtype=torch.bfloat16,
      #   # llm_int8_skip_modules=[]
      # ),
    )

    config_kwargs = dict(
      r=config["adapter_r"],
      alpha=config["adapter_alpha"],
      dropout=config["adapter_dropout"],
      # vera_d=config["adapter_d"],
      # vera_b=config["adapter_b"],
      attn_matrices=["q","k","v"],
      selfattn_lora=True,
      # intermediate_lora=True,
      # output_lora=True,
    )

    base_config = LoRAConfig(
      leave_out=list(range(config["branch_layer"], config["num_layer"])),
      **config_kwargs
    )
    self.model.add_adapter("base", base_config)

    task_config = LoRAConfig(
      leave_out=list(range(config["branch_layer"])),
      **config_kwargs
    )
    for task in config["melt_tasks"]:
      num_labels = config["heads"][task][0]
      self.model.add_adapter(f"task_{task}", task_config)
      self.model.add_classification_head(
        f"task_{task}", num_labels=num_labels, multilabel=task=="target"
      )

    task_adapters = [f"task_{task}" for task in config["melt_tasks"]]
    composition = ac.Stack("base", ac.Parallel(*task_adapters)) # pyright: ignore
    self.model.set_active_adapters(composition)
    self.model.train_adapter(["base", *task_adapters])

    self.norm_layers = [
      v for k,v in self.model.named_parameters()
      if ("layer.8" in k) and "lora" in k
    ]

    print(len([k for k,v in self.model.named_parameters() if v.requires_grad]))

  def forward(self, batches, split):
    metricss, losses, sizes = {}, {}, {}
    for set_name in self.config["flat_datasets"]:
      batch = batches[set_name]

      outputs = self.model(
        input_ids=batch["tokens"],
        attention_mask=batch["mask"].float()
      )

      # TODO dont use tasks in other datasets
      if len(self.config["melt_tasks"]) == 1:
        outputs = [outputs]

      for task_name, output in zip(self.config["melt_tasks"], outputs):
        if task_name in self.config["active_tasks"][set_name]:
          hidden = output.logits
          task = self.tasks[task_name]
          metricss[task_name], losses[task_name] = task.compute(hidden, batch, split)
          sizes[task_name] = batch["size"]

    loss_counts = torch.Tensor(np.concatenate([
      np.full(self.tasks[task_name].loss_dim, batches[set_name]["size"])
      for set_name, task_name in self.config["melt_pairs"]
    ], axis=0)).to(self.model.device)
    loss_weights = loss_counts / loss_counts.mean()

    return metricss, losses, sizes, loss_weights

  def configure_optimizers(self):
    std_params = filter(lambda p: p.requires_grad, self.parameters())
    optimizer = PagedAdamW32bit(params=std_params, lr=self.lr)
    warmup_scheduler = LinearLR(
      optimizer,
      start_factor=0.01,
      end_factor=1.0,
      total_iters=3
    )
    cosine_scheduler = CosineAnnealingLR(
      optimizer,
      T_max=1,
      eta_min=0.1*self.lr,
    )
    scheduler = SequentialLR(
      optimizer,
      schedulers=[warmup_scheduler, cosine_scheduler],
      milestones=[3]
    )
    return [optimizer], [{"scheduler": scheduler, "interval": "epoch"}]

  def state_dict(self, *args, **kwargs):
    state = super().state_dict(*args, **kwargs)
    quant_ends = ('absmax', 'scale', 'quant_map', 'bitsandbytes__nf4')
    return {k: v for k, v in state.items() if not k.endswith(quant_ends)}
