from typing import *
from abc import ABC, abstractmethod

import torch as pt
import adapters as ad
from adapters import composition as ac

from transformers import ElectraModel
from adapters.models.bert.mixin_bert import BertModelAdaptersMixin

from hydra_zen import builds, store, make_config
from .utils import *

class AdapterModel(BertModelAdaptersMixin, ElectraModel): # pyright: ignore
  ...

# TODO fix

class AdapterMethod(ABC):
  @abstractmethod
  def _apply(self, model: AdapterModel) -> None: ...
      
  def apply(self, model: ElectraModel) -> AdapterModel:
    ad.init(model)
    model = cast(AdapterModel, model)
    self._apply(model)
    self.adjust_dtypes(model)
    return model

  @staticmethod
  def load_sources(model: AdapterModel, source_base: str, sources: List[str]) -> None:
    for source in sources:
      model.load_adapter(f"{source_base}/{source}/adapter", load_as=source, with_head=False)

  @staticmethod
  def adjust_dtypes(model): 
    for name, param in model.named_parameters():
      if "adapter" in name or "lora" in name:
        param.data = param.data.to(pt.bfloat16)

class SingleAdapterMethod(AdapterMethod):
  def __init__(self, adapter: ad.AdapterConfig):
    super().__init__()
    self.adapter = adapter

  def _apply(self, model):
    model.add_adapter("adapter", self.adapter)
    model.train_adapter("adapter") # pyright: ignore

class MergeAdapterMethod(AdapterMethod):
  def __init__(self, source_base: str, sources: List[str]):
    self.source_base = source_base
    self.sources = sources

  def _apply(self, model):
    self.load_sources(model, self.source_base, self.sources)
    model.average_adapter(
      adapter_name="merged",
      adapter_list=self.sources, # TODO INCORRECT
      combine_strategy="lora_delta_w_svd",
      svd_rank=8,
    )
    model.train_adapter("merged") # pyright: ignore

class FuseAdapterMethod(AdapterMethod):
  def __init__(self, source_base: str, sources: List[str]):
    self.source_base = source_base
    self.sources = sources

  def _apply(self, model):
    self.load_sources(model, self.source_base, self.sources)
    fusion = ac.Fuse(sources) # pyright: ignore
    model.add_adapter_fusion(fusion)
    model.train_adapter_fusion(fusion)

method_store = store(group="method", to_config=remove_types)
method_store(builds(
  SingleAdapterMethod,
  adapter=builds(
    ad.LoRAConfig,
    r=8,
    alpha=8,
    dropout=0.1,
    dtype="bfloat16",
  )
), name="lora8")
method_store(builds(
  SingleAdapterMethod,
  adapter=builds(
    ad.LoRAConfig,
    r=16,
    alpha=16,
    dropout=0.1,
    dtype="bfloat16",
  )
), name="lora16")
method_store(builds(
  SingleAdapterMethod,
  adapter=builds(
    ad.AdapterPlusConfig,
    reduction_factor=64
  )
), name="aplus64")
method_store(builds(
  SingleAdapterMethod,
  adapter=builds(
    ad.AdapterPlusConfig,
    reduction_factor=16
  )
), name="aplus16")

# class FullModel(BaseModel):
#   def __init__(self, config):
#     model_config = AutoConfig.from_pretrained(config["model"])
#     model_config.hidden_dropout_prob = 0.3
#     model_config.attentions_prob_dropout_prob = 0.3
#     super().__init__(config, model_config)

# class ParallelModel(BaseModel):
#   def __init__(self, config):
#     super().__init__(config)

#     for source in config["merge_sources"]:
#       self.model.load_adapter(f"{config['merge_base']}/{source}/adapter", load_as=source, with_head=False)

#     composition = ac.Parallel(*config["merge_sources"])
#     self.model.set_active_adapters(composition)
#     if config["parallel_train"] == "head":
#       for param in self.model.parameters():
#         param.requires_grad = False
#     elif config["parallel_train"] == "full":
#       self.model.train_adapter(composition)
      
#     self.adjust_dtypes()

#   def forward_base(self, batch):
#     hidden = self.model(
#       input_ids=batch["tokens"],
#       attention_mask=batch["mask"].bfloat16()
#     ).last_hidden_state.float()
#     chunks = []
#     for i in range(len(self.config["merge_sources"])):
#       width = self.config["batch_size"]
#       chunks.append(hidden[width*i: width*(i+1)])
#     return torch.cat(chunks, dim=2)

# class MTLLoraModel(BaseModel):
#   def __init__(self, config):
#     super().__init__(config)

#     adapter_config = MTLLoRAConfig(
#       r=config["adapter_r"],
#       dropout=config["adapter_dropout"],
#       alpha=config["adapter_alpha"],
#       dtype="bfloat16",
#     )
#     for task in config["melt_tasks"]:
#       self.model.add_adapter(task, adapter_config)
#     self.model.share_parameters(adapter_names=config["melt_tasks"])
#     self.model.set_active_adapters(ac.MultiTask(*config["melt_tasks"]))
#     self.model.train_adapter(ac.MultiTask(*config["melt_tasks"]))
#     self.adjust_dtypes()
#     self.vis_params()
  
#   def forward_base(self, batch):
#     return self.model(
#       batch["tokens"],
#       attention_mask=batch["mask"].float(),
#       task_ids=torch.full((batch["size"],), batch["i"]),
#     ).last_hidden_state.float()
    
#   def forward(self, batches, split):
#     metricss, losses, sizes = {}, {}, {}
#     for i, (task_name, batch) in enumerate(batches.items()):
#       hidden = self.forward_base(batch | {"i": i})
#       metricss[task_name], losses[task_name] = self.tasks[task_name].compute(hidden, batch, split)
#       sizes[task_name] = batch["size"]

#     return metricss, losses, sizes

# _model_constructors = {
#   "full": FullModel,
#   "lora": LoraModel,
#   "bn": BottleneckModel,
#   "merge": MergeModel,
#   "fusion": FusionModel,
#   "parallel": ParallelModel,
#   "mtllora": MTLLoraModel,
# }
# def construct_module(cfg):
#   method = cfg.module.method.name
#   if method not in _model_constructors:
#     raise ValueError(f"invalid {method=}")
#   constructor = _model_constructors[method]
#   return constructor(cfg)
