from dataclasses import dataclass

#Structure--------------------------------------------------------------------------------
@dataclass
class Config:
    context: int = 256
    embed: int = 256
    numHeads: int = 4
    numLayers: int = 4
    dropOut: float = 0.1

#Training--------------------------------------------------------------------------------
@dataclass
class Train:
    batch: int = 32
    rate: float = 1e-3
    numSteps: int = 1500
    interval: int = 250