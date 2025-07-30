from typing import *
from dataclasses import is_dataclass

from omegaconf import OmegaConf, DictConfig
from hydra_zen import make_custom_builds_fn
from hydra_zen.wrapper import default_to_config
from hydra_zen import store, zen
from hydra.conf import HydraConf, JobConf
from hydra_zen.typing._implementations import DataClass # bruh

def remove_types(cfg: DictConfig) -> DictConfig:
  cfg = cast(DictConfig, default_to_config(cfg))
  if is_dataclass(cfg):
    cfg = OmegaConf.create(cfg)
    cfg = cast(DictConfig, OmegaConf.to_container(cfg))
    cfg = OmegaConf.create(cfg)
  return cfg

fbuilds = make_custom_builds_fn(populate_full_signature=True)

def run_hydra(method: Callable, config: type[DataClass]) -> None:
  store(HydraConf(job=JobConf(chdir=False)))
  store(config, name="config")
  store.add_to_hydra_store(overwrite_ok=True)
  zen(method).hydra_main(
    config_name="config",
    version_base="1.3",
    config_path=None,
  )
