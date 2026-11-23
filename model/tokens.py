#Vocabulary--------------------------------------------------------------------------------
def build(vocab):
    toId = {ch: i for i, ch in enumerate(vocab)}
    toChar = {i: ch for i, ch in enumerate(vocab)}
    return toId, toChar


#Encoding--------------------------------------------------------------------------------
def encode(text, toId):
    return [toId[ch] for ch in text]


#Decoding--------------------------------------------------------------------------------
def decode(ids, toChar):
    return ''.join([toChar[i] for i in ids])