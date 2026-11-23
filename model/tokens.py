#Import--------------------------------------------------------------------------------
import unicodedata

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


#Filter--------------------------------------------------------------------------------
plain = str.maketrans({"\t": " ", "\xa0": " ", "‘": "'", "’": "'", "“": '"', "”": '"', "–": "-", "—": "-", "…": "..."})
def filter(text):
    text = text.translate(plain)#Translate to plain
    text = unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode()#Strips accents and drops other symbols
    return "".join(c for c in text if c == "\n" or " " <= c <= "~")#Keeps newlines and printable ASCII