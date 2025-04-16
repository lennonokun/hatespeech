import json

def load_stats(config):
  path = config["output_stats_path"].format(name="explain")
  return json.load(open(path, "r"))
