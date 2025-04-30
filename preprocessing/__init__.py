import json

from .explain import do_fix, ExplainPreprocessor
from .measuring import MeasuringPreprocessor

_preprocessors_list = [
  ExplainPreprocessor,
  MeasuringPreprocessor,
]
_preprocessors_dict = {
  x.name: x for x in _preprocessors_list
}

def construct_preprocessor(name, config):
  if name not in _preprocessors_dict:
    raise ValueError(f"invalid preprocessor name: {name}")
  return _preprocessors_dict[name](config)

def load_stats(config):
  path = config["output_stats_path"].format(name="explain")
  return json.load(open(path, "r"))

__all__ = [
  "do_fix",
  "load_stats",
  "construct_preprocessor",
]
