#Import--------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F
from pathlib import Path


#Directory--------------------------------------------------------------------------------
root = Path(__file__).resolve().parent
if not (root / "data").exists():
    root = root.parent

dataRoot = root / "data"
dataFiles = sorted(dataRoot.glob("*.txt"))
tuneRoot = root / "tune" / "gpt.pt"


#Settings--------------------------------------------------------------------------------
context   = 256
embed     = 256
numHeads  = 4
numLayers = 4


batch    = 32
dropout  = 0.1
rate     = 1e-3
numSteps = 1500
interval = 250

temp      = 0.8
prompt    = ""
numTokens = 300

#Device--------------------------------------------------------------------------------
torch.manual_seed(1337)

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"using device: {device}")

class model(nn.Module):
    def __init__(self, context, embed, numHeads, numLayers, batch, dropout, rate, numSteps, interval):
        super().__init__()
        self.context = context
        self.embed = embed
        self.numHeads = numHeads
        self.numLayers = numLayers

    def context():
        query = input("> ")
        return context