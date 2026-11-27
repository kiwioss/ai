#Import--------------------------------------------------------------------------------
from dataclasses import dataclass

#Structure--------------------------------------------------------------------------------
@dataclass
class Config:
    vocab: int = 8192
    context: int = 640
    embed: int = 640
    numHeads: int = 10
    numLayers: int = 10
    dropOut: float = 0.1
    ceiling: int = 100_000

#Training--------------------------------------------------------------------------------
@dataclass
class Train:
    batch: int = 16
    accumulate: int = 4
    rate: float = 3e-4
    interval: int = 100
    ceiling: int = 100_000 #step where the cosine reaches its floor of rate / 10