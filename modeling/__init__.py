from .datamodule import HateDatamodule
from .peft_model import PeftModel
from .std_model import StandardModel
from .hydra_model import HydraModel
from .mtl_model import MTLLoraModel
from .custom import MultiEarlyStopping


_data_methods = {
  "std": "dataset",
  "hydra": "dataset",
  "peft": "dataset",
  "mtllora": "task",
}
def construct_datamodule(config, method=None):
  if method is not None:
    return HateDatamodule(config, method)
  elif config["model_type"] not in _data_methods:
    raise ValueError(f"invalid {config['model_type']=}")

  method = _data_methods[config["model_type"]]
  return HateDatamodule(config, method)

_model_constructors = {
  "std": StandardModel,
  "hydra": HydraModel,
  "peft": PeftModel,
  "mtllora": MTLLoraModel,
}
def construct_module(config):
  if config["model_type"] not in _model_constructors:
    raise ValueError(f"invalid {config['model_type']=}")
  constructor = _model_constructors[config["model_type"]]
  return constructor(config)

__all__ = [
  "construct_datamodule",
  "construct_module",
  "MultiEarlyStopping",
]
