#Import--------------------------------------------------------------------------------
from dataclasses import dataclass

#Structure--------------------------------------------------------------------------------
@dataclass
class Config:
    context: int = 512
    embed: int = 512
    numHeads: int = 8
    numLayers: int = 8
    dropOut: float = 0.15

#Training--------------------------------------------------------------------------------
@dataclass
class Train:
    batch: int = 32
    rate: float = 3e-4
    numSteps: int = 1500
    interval: int = 10