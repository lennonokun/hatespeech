from typing import *
from abc import ABC, abstractmethod

import glob
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
  def adjust_dtypes(model: AdapterModel) -> None:
    for name, param in model.named_parameters():
      if "adapter" in name or "lora" in name or "compacter" in name:
        param.data = param.data.to(pt.bfloat16)

  @staticmethod
  def load_source(model: AdapterModel, source: str) -> None:
    model.load_adapter(f"{source}/encoder/adapter", load_as="adapter", with_head=False)
        
  @staticmethod
  def load_sources(model: AdapterModel, pattern: str) -> List[str]:
    out = []
    for i, source in enumerate(glob.glob(pattern)):
      model.load_adapter(f"{source}/encoder/adapter", load_as=f"source{i}", with_head=False)
      out.append(f"source{i}")
    return out

class LoadAdapterMethod(AdapterMethod):
  def __init__(self, source):
    super().__init__()
    self.source = source

  def _apply(self, model):
    self.load_source(model, self.source)
    model.train_adapter("adapter") # pyright: ignore

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
  def __init__(self, sources: str, svd_rank: int):
    super().__init__()
    print(sources)
    self.sources = sources
    self.svd_rank = svd_rank

  def _apply(self, model):
    adapter_list = self.load_sources(model, self.sources)
    model.average_adapter(
      adapter_name="adapter",
      adapter_list=adapter_list,
      combine_strategy="lora_delta_w_svd",
      svd_rank=self.svd_rank,
    )
    model.train_adapter("adapter") # pyright: ignore

class FuseAdapterMethod(AdapterMethod):
  def __init__(self, sources: str):
    super().__init__()
    self.sources = sources

  def _apply(self, model):
    adapter_list = self.load_sources(model, self.sources)
    fusion = ac.Fuse(adapter_list) # pyright: ignore
    model.add_adapter_fusion(fusion)
    model.train_adapter_fusion(fusion)

method_store = store(group="method", to_config=remove_types)
method_store(builds(NoAdapterMethod), name="full")
method_store(builds(LoadAdapterMethod, source="${load_path}"), name="ah_load")
  
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
  sources=MISSING
), name="ah_fuse")
