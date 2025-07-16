from typing import *
from abc import ABC, abstractmethod

import torch as pt
import adapters as ad
from adapters import composition as ac
import peft as pe

from transformers import ElectraModel
from adapters.models.bert.mixin_bert import BertModelAdaptersMixin

from hydra_zen import builds, store
from omegaconf import MISSING
from .utils import *

class AdapterModel(BertModelAdaptersMixin, ElectraModel): # pyright: ignore
  ...

class AdapterMethod(ABC):
  def __init__(self):
    super().__init__()
  
  @abstractmethod
  def _apply(self, model: AdapterModel) -> None: ...
      
  def apply(self, model: ElectraModel) -> AdapterModel:
    ad.init(model)
    model = cast(AdapterModel, model)
    self._apply(model)
    self.adjust_dtypes(model)
    return model

  @staticmethod
  def adjust_dtypes(model): 
    for name, param in model.named_parameters():
      if "adapter" in name or "lora" in name or "compacter" in name:
        param.data = param.data.to(pt.bfloat16)

  @staticmethod
  def load_sources(model: AdapterModel, source_base: str, sources: List[str]) -> None:
    for source in sources:
      model.load_adapter(f"{source_base}/{source}/encoder/adapter", load_as=source, with_head=False)

class LoadAdapterMethod(AdapterMethod):
  def __init__(self, path):
    super().__init__()
    self.path = path

  def _apply(self, model):
    model.load_adapter(f"{self.path}/encoder/adapter", load_as="adapter", with_head=False)
    model.train_adapter("adapter")

class NoAdapterMethod(AdapterMethod):
  def __init__(self):
    super().__init__()

  def _apply(self, model):
    pass

class PeftAdapterMethod(AdapterMethod):
  def __init__(self, adapter: pe.PeftConfig):
    super().__init__()
    self.adapter = adapter

  # TODO not best practice
  def apply(self, model):
    model = pe.get_peft_model(model, self.adapter)
    model.train()
    self.adjust_dtypes(model)
    return model

  def _apply(self, model):
    pass

class SingleAdapterMethod(AdapterMethod):
  def __init__(self, adapter: ad.AdapterConfig):
    super().__init__()
    self.adapter = adapter

  def _apply(self, model):
    model.add_adapter("adapter", self.adapter)
    model.train_adapter("adapter") # pyright: ignore

class MergeAdapterMethod(AdapterMethod):
  def __init__(self, source_base: str, sources: List[str], svd_rank: int):
    super().__init__()
    self.source_base = source_base
    self.sources = sources
    self.svd_rank = svd_rank

  def _apply(self, model):
    self.load_sources(model, self.source_base, self.sources)
    model.average_adapter(
      adapter_name="merged",
      adapter_list=self.sources, # TODO INCORRECT
      combine_strategy="lora_delta_w_svd",
      svd_rank=self.svd_rank,
    )
    model.train_adapter("merged") # pyright: ignore

class FuseAdapterMethod(AdapterMethod):
  def __init__(self, source_base: str, sources: List[str]):
    super().__init__()
    self.source_base = source_base
    self.sources = sources

  def _apply(self, model):
    self.load_sources(model, self.source_base, self.sources)
    fusion = ac.Fuse(sources) # pyright: ignore
    model.add_adapter_fusion(fusion)
    model.train_adapter_fusion(fusion)

method_store = store(group="method", to_config=remove_types)
method_store(builds(NoAdapterMethod), name="full")
method_store(builds(LoadAdapterMethod, path="${load_path}"), name="ah_load")
  
for r in [8, 16, 24, 32]:
  method_store(builds(
    PeftAdapterMethod,
    adapter=builds(
      pe.LoraConfig,
      task_type=pe.TaskType.FEATURE_EXTRACTION,
      inference_mode=False,
      r=r,
      lora_alpha=r,
      lora_dropout=0.1,
      target_modules=["query", "key"],
      use_rslora=True,
    )
  ), name=f"pe_lora{r}")
  method_store(builds(
    SingleAdapterMethod,
    adapter=builds(
      ad.LoRAConfig,
      r=r,
      alpha=r,
      dropout=0.1,
      dtype="bfloat16",
    )
  ), name=f"ah_lora{r}")
  method_store(builds(
    MergeAdapterMethod,
    source_base=MISSING,
    sources=MISSING,
    svd_rank=r,
  ), name=f"ah_merge{r}")

method_store(builds(
  SingleAdapterMethod,
  adapter=builds(
    ad.CompacterPlusPlusConfig,
  )
), name=f"ah_cpp")
method_store(builds(
  SingleAdapterMethod,
  adapter=builds(
    ad.PrefixTuningConfig,
  )
), name=f"ah_prefix")
  
for f in [64, 32, 16]:
  method_store(builds(
    SingleAdapterMethod,
    adapter=builds(
      ad.AdapterPlusConfig,
      reduction_factor=f
    )
  ), name=f"ah_aplus{f}")
method_store(builds(
  FuseAdapterMethod,
  source_base=MISSING,
  sources=MISSING,
), name="ah_fuse")
