#builds the character to id and id to character dictionaries, encodes text to ids, and decodes ids back to text
def build(vocab):
    toId = {ch: i for i, ch in enumerate(vocab)}
    toChar = {i: ch for i, ch in enumerate(vocab)}
    return toId, toChar

def encode(text, toId):
    return [toId[ch] for ch in text]


def decode(ids, toChar):
    return ''.join([toChar[i] for i in ids])