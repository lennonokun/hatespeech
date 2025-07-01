from pydantic import BaseModel

from .utils import fbuilds

class HateDebug(BaseModel):
  vis_params: bool

HateDebugCfg = fbuilds(
  HateDebug,
  vis_params=False
)
