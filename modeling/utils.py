from typing import *
from dataclasses import is_dataclass

from omegaconf import OmegaConf, DictConfig
from hydra_zen import make_custom_builds_fn
from hydra_zen.wrapper import default_to_config

def remove_types(cfg: DictConfig) -> DictConfig:
  cfg = cast(DictConfig, default_to_config(cfg))
  if is_dataclass(cfg):
    cfg = OmegaConf.create(cfg)
    cfg = cast(DictConfig, OmegaConf.to_container(cfg))
    cfg = OmegaConf.create(cfg)
  return cfg

fbuilds = make_custom_builds_fn(populate_full_signature=True)
