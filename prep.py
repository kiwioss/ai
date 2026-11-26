#Import--------------------------------------------------------------------------------
from tokenizers import Tokenizer, models, trainers, pre_tokenizers, decoders
from pathlib import Path
from tqdm import tqdm
import numpy as np

from model.config import Config


#Paths--------------------------------------------------------------------------------
root = Path(__file__).resolve().parent
dataDir = root / "data"
stateDir = root / "state"
vocabDir = stateDir / "token.json"
sampleDir = stateDir / "sample.txt"


#Settings--------------------------------------------------------------------------------
sampleMB = 400
weights = {"gn": 0.5, "lw": 0.5, "ch": 2} #change to prefix of file
batchDocs = 10_000


#Sample--------------------------------------------------------------------------------
def buildSample():
    sections = {}
    for f in sorted(dataDir.glob("*.txt")): #Find all txt
        sections.setdefault(f.name[:2], []).append(f) 

    weighted = {c: sum(f.stat().st_size for f in fs) * weights.get(c, 1) for c, fs in sections.items()}
    total = sum(weighted.values())

    with open(sampleDir, "w", encoding="utf-8") as out:
        for code, files in sections.items():
            share = sampleMB * weighted[code] / total #This section's MB in the sample
            perFile = share / len(files)
            for f in files:
                with open(f, encoding="utf-8") as src:
                    out.write(src.read(int(perFile * 1_000_000)) + "\n\n") #Only the first part of each file
            print(f"  {code}  {share:6.1f} MB")


#Tokenizer--------------------------------------------------------------------------------
def trainTokenizer(vocab):
    tokenizer = Tokenizer(models.BPE())
    tokenizer.pre_tokenizer = pre_tokenizers.Sequence([
        pre_tokenizers.Digits(individual_digits=True), #2026 → 2 0 2 6
        pre_tokenizers.ByteLevel(add_prefix_space=False), #Bytes, so any character works
    ])
    tokenizer.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(
        vocab_size=vocab,
        special_tokens=["<|eot|>"],
        initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), #256 bytes always in the vocab
        show_progress=True,
    )
    tokenizer.train([str(sampleDir)], trainer)
    tokenizer.save(str(vocabDir))
    sampleDir.unlink() #Remove Sample.txt


#Encode--------------------------------------------------------------------------------
def readDocs(f):
    lines = []
    for line in f:
        if line.strip() == "": #A blank line ends a document
            if lines:
                yield "".join(lines).rstrip("\n")
                lines = []
        else:
            lines.append(line)
    if lines:
        yield "".join(lines).rstrip("\n")

def writeBatch(tokenizer, docs, eot, out):
    ids = []
    for enc in tokenizer.encode_batch(docs): #Uses every core
        ids.extend(enc.ids)
        ids.append(eot) #Marks the end of each document
    np.array(ids, dtype=np.uint16).tofile(out)
    return len(ids)

def encodeFile(tokenizer, txtPath):
    eot = tokenizer.token_to_id("<|eot|>")
    tokens, chars, docs = 0, 0, []
    with open(txtPath, encoding="utf-8") as f, open(txtPath.with_suffix(".bin"), "wb") as out:
        for doc in readDocs(f):
            docs.append(doc)
            chars += len(doc)
            if len(docs) == batchDocs:
                tokens += writeBatch(tokenizer, docs, eot, out)
                docs = []
        if docs:
            tokens += writeBatch(tokenizer, docs, eot, out)
    return tokens, chars


#Run--------------------------------------------------------------------------------
if __name__ == "__main__":
    stateDir.mkdir(exist_ok=True)
    vocab = Config().vocab

    if not vocabDir.exists(): #Only the first run trains a tokenizer
        print(f"building {sampleMB} MB sample")
        buildSample()
        print(f"translating vocab {vocab:,}")
        trainTokenizer(vocab)

    tokenizer = Tokenizer.from_file(str(vocabDir))
    if tokenizer.get_vocab_size() != vocab:
        raise SystemExit(f"tokenizer and vocab in config differ")

    files = sorted(dataDir.glob("*.txt"))
    totalTokens, totalChars = 0, 0
    for txt in tqdm(files, desc="encoding"):
        binPath = txt.with_suffix(".bin")
        if binPath.exists() and binPath.stat().st_mtime > txt.stat().st_mtime: #Already done
            continue
        tokens, chars = encodeFile(tokenizer, txt)
        totalTokens += tokens
        totalChars += chars
        tqdm.write(f"{binPath.name} | {tokens:,} tokens | {chars / max(tokens, 1):.2f} chars per token")

    allTokens = sum(b.stat().st_size // 2 for b in dataDir.glob("*.bin")) #2 bytes per token
    print(f"{allTokens:,} tokens in data/*.bin ({allTokens * 2 / 1e9:.2f} GB)")
    if totalTokens:
        print(f"{totalChars / totalTokens:.2f} chars per token")