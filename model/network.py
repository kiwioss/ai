#Import---------------------------------------------------------------------------------------
import torch
import torch.nn as nn
import torch.nn.functional as F


#RoPE-----------------------------------------------------------------------------------------
#turns each pair of numbers by an angle so the score only sees how far apart two tokens are
def spin(x, cos, sin):
    a, b = x.chunk(2, dim=-1) #first half and second half so number i pairs with number i plus 32
    return torch.cat((a * cos - b * sin, a * sin + b * cos), dim=-1) #turn every pair by its own angle


#SelfAttention--------------------------------------------------------------------------------
#every token looks back at earlier tokens and copies in information from the ones that matter to it
class SelfAttention(nn.Module):
    def __init__(self, embed, numHeads, dropOut):
        super().__init__()

        if embed % numHeads != 0: #each head needs a whole number of values
            raise ValueError(f"embed {embed} must divide evenly by numHeads {numHeads}") #for example 1024 over 12 is 85.33 and crashes

        self.numHeads = numHeads
        headDim = embed // numHeads #64 numbers per head which makes 32 pairs
        self.register_buffer("speeds", 1.0 / 10000 ** (torch.arange(0, headDim, 2).float() / headDim), persistent=False) #spin speed of each pair from fast to slow and a buffer moves with to device but is never learned or saved

        self.prism = nn.Linear(embed, 3 * embed) #splits each token into its query key and value in one multiply
        self.out = nn.Linear(embed, embed) #mixes what all heads found into one result per token
        self.dropOut = dropOut #attention dropout used only while training
        self.outDrop = nn.Dropout(dropOut) #dropout on the final result

    def forward(self, vectors, cache = None, begin = 0):
        B, T, E = vectors.shape #batch size then tokens then numbers per token

        q, k, v = self.prism(vectors).split(E, dim=2) #query is what a token looks for and key is what it offers and value is what it hands over
        q, k, v = [t.view(B, T, self.numHeads, E // self.numHeads).transpose(1, 2) for t in (q, k, v)] #cut into heads so each head searches on its own

        angles = torch.outer(torch.arange(begin, begin + T, device=vectors.device).float(), self.speeds) #real positions of the new tokens which start at begin instead of 0 when a cache is used
        cos, sin = angles.cos().to(q.dtype), angles.sin().to(q.dtype) #same dtype as q so bf16 does not clash
        q, k = spin(q, cos, sin), spin(k, cos, sin) #only q and k turn because v is what gets handed over not what gets matched

        if cache is not None: #glue the keys and values saved from earlier steps in front of the new ones
            k = torch.cat((cache[0], k), dim=2)
            v = torch.cat((cache[1], v), dim=2)

        result = F.scaled_dot_product_attention(q, k, v, is_causal=cache is None, dropout_p=self.dropOut if self.training else 0.0) #mask only without a cache since one new token is never ahead of anything and is_causal would hide all but the first key

        result = result.transpose(1, 2).contiguous().view(B, T, E) #glue the heads back into one vector per token
        return self.outDrop(self.out(result)), (k, v) #the result plus this layer keys and values for the next step


#FeedForward----------------------------------------------------------------------------------
#works on each token alone and is where most learned facts are stored
class FeedForward(nn.Module):
    def __init__(self, embed, dropOut):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(embed, 4 * embed), #widen to 4 times the size to give room to compute
            nn.GELU(), #bends the values so the network can learn more than straight lines
            nn.Linear(4 * embed, embed), #shrink back to the normal size
            nn.Dropout(dropOut)) #randomly drops some values while training so it does not memorize

    def forward(self, vectors):
        return self.net(vectors)


#Block----------------------------------------------------------------------------------------
#one layer of the model which is attention then feed forward each added onto the input
class Block(nn.Module):
    def __init__(self, embed, numHeads, dropOut):
        super().__init__()

        self.normalOne = nn.LayerNorm(embed) #keeps values at a steady size before attention
        self.attention = SelfAttention(embed, numHeads, dropOut)
        self.normalTwo = nn.LayerNorm(embed) #keeps values at a steady size before feed forward
        self.feedForward = FeedForward(embed, dropOut)

    def forward(self, vectors, cache = None, begin = 0):
        change, cache = self.attention(self.normalOne(vectors), cache, begin) #attention now hands back its change and its keys and values
        vectors = vectors + change #add instead of replace so each block only adds a change
        vectors = vectors + self.feedForward(self.normalTwo(vectors))
        return vectors, cache


#Whole Model----------------------------------------------------------------------------------
class Machine(nn.Module):
    def __init__(self, vocab, context, embed, numHeads, numLayers, dropOut):
        super().__init__()

        self.context = context #longest window the model was trained on and query cuts to it
        self.tokenEmbed = nn.Embedding(vocab, embed) #one learned vector per token
        self.blocks = nn.Sequential(*[Block(embed, numHeads, dropOut) for _ in range(numLayers)]) #numLayers blocks run one after another
        self.norm = nn.LayerNorm(embed) #final steadying before the output
        self.head = nn.Linear(embed, vocab) #a score for every token in the vocab
        self.head.weight = self.tokenEmbed.weight #shares the embedding table so input and output meanings match and saves vocab times embed params

        self.apply(self.initWeight) #small random starting weights

        #layers that add into the residual stream start even smaller so deep stacks stay stable and the first loss lands near ln of vocab
        for name, p in self.named_parameters():
            if name.endswith("attention.out.weight") or name.endswith("feedForward.net.2.weight"):
                nn.init.normal_(p, std=0.02 / (2 * numLayers) ** 0.5)

    def initWeight(self, module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
        if isinstance(module, nn.Linear) and module.bias is not None:
            nn.init.zeros_(module.bias)

    def forward(self, ids, targets=None, caches=None, begin=0):
        vectors = self.tokenEmbed(ids) #only what each token is since where it is now comes from spin inside attention

        for i, block in enumerate(self.blocks): #a loop instead of self.blocks(vectors) because each block now passes a cache too
            vectors, new = block(vectors, caches[i] if caches is not None else None, begin)
            if caches is not None:
                caches[i] = new #save this layer keys and values in the list the caller gave so training stays unchanged

        vectors = self.norm(vectors)
        logits = self.head(vectors) #raw scores for the next token at every position

        loss = None
        if targets is not None:
            loss = F.cross_entropy(logits.view(-1, logits.size(-1)), targets.view(-1)) #how surprised the model was by the real next tokens

        return logits, loss

    @torch.no_grad() #generating only with no gradients
    def query(self, ids, count, temp = 0.75, stop=None, topK=50): #temp below 0.01 is raised to 0.01 so near greedy still works
        temp = max(temp, 0.01) #anything below 0.01 becomes 0.01 so dividing by temp is always safe

        caches = [None] * len(self.blocks) #one empty spot per layer
        logits, _ = self(ids[:, -self.context:], caches=caches) #prefill reads the whole prompt once and fills every layer cache
        begin = min(ids.size(1), self.context) #position of the next new token
        finished = torch.zeros(ids.size(0), dtype=torch.bool, device=ids.device) #one done flag per row so batches work

        for _ in range(count):
            logits = logits[:, -1, :] / temp #only the last position predicts the next token and temp sharpens or flattens it

            if topK:
                cutoff = torch.topk(logits, topK).values[:, -1:] #score of the topK best token
                logits = logits.masked_fill(logits < cutoff, float("-inf")) #everything worse gets 0 chance

            probs = F.softmax(logits, dim=-1) #scores into chances that add up to 1
            nextId = torch.multinomial(probs, num_samples=1) #pick one token by those chances

            if stop is not None:
                nextId[finished] = stop #rows that already ended keep writing stop
                finished |= nextId.squeeze(1) == stop #mark rows that just hit stop

            ids = torch.cat((ids, nextId), dim=1) #add it to the text

            if stop is not None and finished.all(): #every row is done
                break

            caches[:] = [(k[:, :, -(self.context - 1):], v[:, :, -(self.context - 1):]) for k, v in caches] #keep at most context minus 1 old tokens so the window stays at context
            logits, _ = self(nextId, caches=caches, begin=begin) #only the new token goes in and the cache supplies the rest
            begin += 1 #rope only sees gaps so this can grow past context safely

        return ids