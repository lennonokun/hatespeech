from hydra_zen import make_config, store
from .utils import remove_types

experiment_store = store(group="experiment", package="_global_", to_config=remove_types)
experiment_store(make_config(
  hydra_defaults=[
    {"override /tasks": ["target"]},
    {"override /quantization": "none"},
    {"override /optimization": "fast"},
    {"override /method": "ah_lora16"},
    "_self_",
  ],
  datamodule=dict(
    batch_size=200,
  )
), name="simple_target")
experiment_store(make_config(
  hydra_defaults=[
    {"override /tasks": ["label"]},
    {"override /quantization": "none"},
    {"override /optimization": "fast"},
    {"override /method": "ah_lora16"},
    "_self_",
  ],
  datamodule=dict(
    batch_size=200,
  )
), name="simple_label")
