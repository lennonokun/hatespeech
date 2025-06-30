from hydra_zen import make_config, store
from .utils import remove_types

SimpleTargetExperiment = make_config(
  hydra_defaults=[
    {"override /tasks": ["target"]},
    {"override /quantization": "none"},
    {"override /optimization": "fast"},
    {"override /method": "lora16"},
    "_self_",
  ],
  datamodule=dict(
    batch_size=150,
  )
)

experiment_store = store(group="experiment", package="_global_", to_config=remove_types)
experiment_store(SimpleTargetExperiment, name="simple_target")
