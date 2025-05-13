import torch

from transformers import AutoConfig
from adapters import *
from adapters import composition as ac

from .base_model import BaseModel

class FullModel(BaseModel):
  def __init__(self, config):
    model_config = AutoConfig.from_pretrained(config["model"])
    model_config.hidden_dropout_prob = 0.3
    model_config.attentions_prob_dropout_prob = 0.3
    super().__init__(config, model_config)

class LoraModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)
    lora_config = LoRAConfig(
      r=config["adapter_r"],
      alpha=config["adapter_alpha"],
      dropout=config["adapter_dropout"],
      # attn_matrices=["q","k","v"],
      # selfattn_lora=True,
      # intermediate_lora=True,
      # output_lora=True,
      dtype="bfloat16",
    )
    self.model.add_adapter("adapter", lora_config)
    self.model.set_active_adapters("adapter")
    self.model.train_adapter("adapter")
    self.adjust_dtypes()

class BottleneckModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)
    aplus_config = AdapterPlusConfig(reduction_factor=64)
    self.model.add_adapter("adapter", aplus_config)
    self.model.set_active_adapters("adapter")
    self.model.train_adapter("adapter")
    self.adjust_dtypes()

class MergeModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)

    for source in config["merge_sources"]:
      self.model.load_adapter(f"{config['merge_base']}/{source}/adapter", load_as=source, with_head=False)

    self.model.average_adapter(
      adapter_name="qlora",
      adapter_list=config["melt_tasks"],
      combine_strategy="lora_delta_w_svd",
      svd_rank=8,
    )
    self.model.train_adapter("qlora")
    self.adjust_dtypes()

class FusionModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)

    for source in config["merge_sources"]:
      self.model.load_adapter(f"{config['merge_base']}/{source}/adapter", load_as=source, with_head=False)

    fusion = ac.Fuse(*config["merge_sources"])
    self.model.add_adapter_fusion(fusion)
    self.model.train_adapter_fusion(fusion)
    self.adjust_dtypes()

class ParallelModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)

    for source in config["merge_sources"]:
      self.model.load_adapter(f"{config['merge_base']}/{source}/adapter", load_as=source, with_head=False)

    composition = ac.Parallel(*config["merge_sources"])
    self.model.set_active_adapters(composition)
    if config["parallel_train"] == "head":
      for param in self.model.parameters():
        param.requires_grad = False
    elif config["parallel_train"] == "full":
      self.model.train_adapter(composition)
      
    self.adjust_dtypes()

  def forward_base(self, batch):
    hidden = self.model(
      input_ids=batch["tokens"],
      attention_mask=batch["mask"].bfloat16()
    ).last_hidden_state.float()
    chunks = []
    for i in range(len(self.config["merge_sources"])):
      width = self.config["batch_size"]
      chunks.append(hidden[width*i: width*(i+1)])
    return torch.cat(chunks, dim=2)

class MTLLoraModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)

    adapter_config = MTLLoRAConfig(
      r=config["adapter_r"],
      dropout=config["adapter_dropout"],
      alpha=config["adapter_alpha"],
      dtype="bfloat16",
    )
    for task in config["melt_tasks"]:
      self.model.add_adapter(task, adapter_config)
    self.model.share_parameters(adapter_names=config["melt_tasks"])
    self.model.set_active_adapters(ac.MultiTask(*config["melt_tasks"]))
    self.model.train_adapter(ac.MultiTask(*config["melt_tasks"]))
    self.adjust_dtypes()
    self.vis_params()
  
  def forward_base(self, batch):
    return self.model(
      batch["tokens"],
      attention_mask=batch["mask"].float(),
      task_ids=torch.full((batch["size"],), batch["i"]),
    ).last_hidden_state.float()
    
  def forward(self, batches, split):
    metricss, losses, sizes = {}, {}, {}
    for i, (task_name, batch) in enumerate(batches.items()):
      hidden = self.forward_base(batch | {"i": i})
      metricss[task_name], losses[task_name] = self.tasks[task_name].compute(hidden, batch, split)
      sizes[task_name] = batch["size"]

    return metricss, losses, sizes

_model_constructors = {
  "full": FullModel,
  "lora": LoraModel,
  "bn": BottleneckModel,
  "merge": MergeModel,
  "fusion": FusionModel,
  "parallel": ParallelModel,
  "mtllora": MTLLoraModel,
}
def construct_module(config):
  if config["model_type"] not in _model_constructors:
    raise ValueError(f"invalid {config['model_type']=}")
  constructor = _model_constructors[config["model_type"]]
  return constructor(config)
