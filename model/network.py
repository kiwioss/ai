#Import---------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


#SelfAttention--------------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, context, embed, numHeads, dropOut):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed, numHeads, dropout=dropOut) #Attention Layer
        mask = torch.tril(torch.ones(context, context)).view(1, 1, context, context) #Mask for causal attention & Hidden Answers
        self.register_buffer('mask', mask) #Register the mask as a buffer

    def forward(self,vectors):
        length = vectors.size(0) #Get the length of the input sequence
        result, _ = self.attention(vectors, vectors, vectors, attn_mask=self.mask[:length,:length], need_weights=False) #Apply attention with the mask
        return result


#FeedForward----------------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, embed, dropOut):
        super().__init__()
        self.embed = embed
        self.dropOut = dropOut


#Block----------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, embed, numHeads, dropOut):
        super().__init__()
        self.embed = embed
        self.numHeads = numHeads
        self.dropOut = dropOut


#Whole Model----------------------------------------------------------------------------------
class Machine(nn.Module):
    def __init__(self, device, context, embed, numHeads, numLayers, dropOut):
        super().__init__()
        self.device = device
        self.context = context
        self.embed = embed
        self.numHeads = numHeads
        self.numLayers = numLayers
        self.dropOut = dropOut

    def query(self, text):
        return self.context