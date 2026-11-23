#Import---------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


#SelfAttention--------------------------------------------------------------------------------
class SelfAttention(nn.Module):
    def __init__(self, context, embed, numHeads, dropOut):
        super().__init__()
        self.attention = nn.MultiheadAttention(embed, numHeads, dropout=dropOut, batch_first=True) #Attention Layer
        mask = torch.triu(torch.ones(context, context), diagonal=1).bool() #Mask for causal attention
        self.register_buffer('mask', mask) #Register the mask as a buffer

    def forward(self,vectors):
        length = vectors.size(1) #Get the length of the input sequence
        result, _ = self.attention(vectors, vectors, vectors, attn_mask=self.mask[:length,:length], need_weights=False) #Apply attention with the mask
        return result


#FeedForward----------------------------------------------------------------------------------
class FeedForward(nn.Module):
    def __init__(self, embed, dropOut):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(embed, 4 * embed), #Expand the embedding dimension
            nn.GELU(), #Activation function
            nn.Linear(4 * embed, embed), #Project back to the original embedding dimension
            nn.Dropout(dropOut)) #Dropout for regularization

    def forward(self, vectors):
        return self.net(vectors) #Apply the feedforward network


#Block----------------------------------------------------------------------------------------
class Block(nn.Module):
    def __init__(self, context, embed, numHeads, dropOut):
        super().__init__()
        self.normalOne = nn.LayerNorm(embed) #First layer normalization
        self.attention = SelfAttention(context, embed, numHeads, dropOut) #Self-attention layer
        self.normalTwo = nn.LayerNorm(embed) #Second layer normalization
        self.feedForward = FeedForward(embed, dropOut) #Feedforward layer

    def forward(self, vectors):
        vectors = vectors + self.attention(self.normalOne(vectors)) #Residual connection with attention
        vectors = vectors + self.feedForward(self.normalTwo(vectors)) #Residual connection with feedforward
        return vectors


#Whole Model----------------------------------------------------------------------------------
class Machine(nn.Module):
    def __init__(self, numChars, context, embed, numHeads, numLayers, dropOut):
        super().__init__()
        self.context = context
        self.tokenEmbed = nn.Embedding(numChars, embed) #Token embedding layer
        self.posEmbed = nn.Embedding(context, embed) #Position embedding layer
        self.blocks = nn.Sequential(*[Block(context, embed, numHeads, dropOut) for _ in range(numLayers)]) #Stack of blocks 🚨
        self.norm = nn.LayerNorm(embed) #Final layer normalization
        self.head = nn.Linear(embed, numChars) #Output layer projecting to number of characters
        self.head.weight = self.tokenEmbed.weight #Tie the weights of the output layer to the token embedding layer

    def forward(self, ids, targets=None):
        length = ids.size(1) #Get the length of the input sequence
        positions = torch.arange(length, device=ids.device).unsqueeze(0) #Create position indices
        vectors = self.tokenEmbed(ids) + self.posEmbed(positions) #Combine token and position embeddings
        vectors = self.blocks(vectors) #Pass through the stack of blocks
        vectors = self.norm(vectors) #Apply final layer normalization
        logits = self.head(vectors) #Project to output logits

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #Compute cross-entropy loss

        return logits, loss #Return logits and loss

    @torch.no_grad()
    def query(self, ids, count, temp = 0.5, stop=None):
        for _ in range(count):
            window = ids[:, -self.context:] #Only the last 'context' tokens go into the model, ids stays whole 🚨
            logits, _ = self(window) #Get logits from the model 🚨
            logits = logits[:, -1, :] / temp #Focus on the last token and apply temperature scaling
            probs = F.softmax(logits, dim=-1) #Convert logits to probabilities
            next_id = torch.multinomial(probs, num_samples=1) #Sample the next token ID
            ids = torch.cat((ids, next_id), dim=1) #Append the new token ID to the sequence

            if stop is not None and next_id.item() == stop: #Check for stop condition
                break

        return ids