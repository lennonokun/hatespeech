import torch

from adapters import LoRAConfig, BnConfig, AdapterPlusConfig, MTLLoRAConfig
from adapters import composition as ac

from .base_model import BaseModel

class StandardModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)
    # adapter_config = BnConfig(
    #   mh_adapter=True,
    #   output_adapter=True,
    #   reduction_factor=16,
    #   non_linearity="relu"
    # )
    # lora_config = LoRAConfig(
    #   r=config["adapter_r"],
    #   alpha=config["adapter_alpha"],
    #   dropout=config["adapter_dropout"],
    #   attn_matrices=["q","k","v"],
    #   selfattn_lora=True,
    #   # intermediate_lora=True,
    #   # output_lora=True,
    # )
    self.model.add_adapter("adapter", AdapterPlusConfig(reduction_factor=16))
    self.model.train_adapter("adapter")
    self.adjust_dtypes()
    # self.norm_layers = [
    #   v for k,v in self.model.named_parameters()
    #   if ("layer.8" in k) and "lora" in k
    # ]

    # print([(k,v.dtype) for k,v in self.model.named_parameters() if v.requires_grad])
  
  def save(self, adapter_dir, head_dir):
    if adapter_dir is not None:
      self.model.save_all_adapters(adapter_dir)
    if head_dir is not None:
      self.model.save_all_heads(head_dir)

class FusionModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)

    for task in config["melt_tasks"]:
      self.model.load_adapter(f"adapters/bn_{task}/adapter", load_as=task, with_head=False)
    # self.model.average_adapter(
    #   adapter_name="qlora",
    #   adapter_list=task_adapters,
    #   combine_strategy="lora_delta_w_svd",
    #   svd_rank=8,
    # )
    fusion = ac.Fuse(*config["melt_tasks"])
    self.model.add_adapter_fusion(fusion)
    self.model.train_adapter_fusion(fusion)
    self.adjust_dtypes()

    # self.norm_layers = [
    #   v for k,v in self.model.named_parameters()
    #   if ("layer.8" in k) and "lora" in k
    # ]

class HydraModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)

    config_kwargs = {
      "r": config["adapter_r"],
      "alpha": config["adapter_alpha"],
      "dropout": config["adapter_dropout"],
      # "attn_matrices": ["q","k","v"],
      # "selfattn_lora": True,
    }

    # TODO: ConfigUnion?
    base_exclude = list(range(config["branch_layer"], config["num_layer"]))
    base_config = LoRAConfig(leave_out=base_exclude, **config_kwargs)
    self.model.add_adapter("base", base_config)

    task_exclude = list(range(config["branch_layer"]))
    task_config = LoRAConfig(leave_out=task_exclude, **config_kwargs)
    for task in config["melt_tasks"]:
      self.model.add_adapter(task, task_config)

    composition = ac.Stack("base", ac.Parallel(*config["melt_tasks"])) # pyright: ignore
    self.model.active_adapters = composition
    self.model.train_adapter(["base", *config["melt_tasks"]])

    self.adjust_dtypes()

    # self.norm_layers = [
    #   v for k,v in self.model.named_parameters()
    #   if ("layer.8" in k) and "lora" in k
    # ]

class MTLLoraModel(BaseModel):
  def __init__(self, config):
    super().__init__(config)

    adapter_config = MTLLoRAConfig(
      r=config["adapter_r"],
      dropout=config["adapter_dropout"],
      alpha= config["adapter_alpha"],
    )
    for task in config["melt_tasks"]:
      self.model.add_adapter(task, adapter_config)
    self.model.share_parameters(adapter_names=config["melt_tasks"])
    self.model.set_active_adapters(ac.MultiTask(*config["melt_tasks"]))
    self.model.train_adapter(ac.MultiTask(*config["melt_tasks"]))
    self.adjust_dtypes()

  def forward_encoder(self, batch, i):
    return self.model(
      batch["tokens"],
      attention_mask=batch["mask"].float(),
      task_ids=torch.full((batch["size"],), i),
    )
  
  def forward(self, batches, split):
    metricss, losses, sizes = {}, {}, {}
    for i, (task_name, batch) in enumerate(batches.items()):
      output = self.forward_encoder(batch, i).head_outputs[i]
      metricss[task_name], losses[task_name] = self.tasks[task_name].compute(output, batch, split)
      sizes[task_name] = batch["size"]

    return metricss, losses, sizes

_model_constructors = {
  "std": StandardModel,
  "hydra": HydraModel,
  "fusion": FusionModel,
  "mtllora": MTLLoraModel,
}
def construct_module(config):
  if config["model_type"] not in _model_constructors:
    raise ValueError(f"invalid {config['model_type']=}")
  constructor = _model_constructors[config["model_type"]]
  return constructor(config)
