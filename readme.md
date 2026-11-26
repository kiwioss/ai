README · MD


# KiwiML
 
A small language model built and trained from scratch: byte-level BPE tokenizer, decoder-only transformer, trained on roughly 3 billion tokens of human-written text.
 
---
 
## Contents
 
1. [Install](#install)
2. [Project layout](#project-layout)
3. [Quick start](#quick-start)
4. [Data naming convention](#data-naming-convention)
5. [Tokenizer](#tokenizer)
6. [Model properties](#model-properties)
7. [Training settings](#training-settings)
8. [Weights and checkpoints](#weights-and-checkpoints)
9. [Data sources](#data-sources)
10. [Model history](#model-history)
---
 
## Install
 
Requirements:
 
- macOS with Apple Silicon (trains on the GPU through MPS), or Linux/Windows with an NVIDIA GPU (CUDA). CPU works but is very slow.
- Python 3.12 or newer
- About 20 GB free disk space (text + token files + checkpoints)
- 16 GB RAM or more
```
cd ~/Desktop/kiwi/ai
python3 -m venv .venv
.venv/bin/pip install torch tokenizers numpy tqdm pyarrow
```
 
Check that the GPU is found:
 
```
.venv/bin/python -c "import torch; print(torch.backends.mps.is_available() or torch.cuda.is_available())"
```
 
`True` means training will use the GPU.
 
---
 
## Project layout
 
```
ai/
├── model/              code that gets imported
│   ├── config.py       model shape (Config) and training settings (Train)
│   ├── network.py      the transformer (Machine)
│   └── tokens.py       old character tokenizer (KiwiML v1 to v4 only)
├── data/               training data
│   ├── xx000001.txt    source text, one file per dataset
│   ├── xx000001.bin    the same text as token ids (made by translate.py)
│   ├── naming.csv      the 26 section codes in priority order
│   └── sources.csv     which dataset each file came from, with size
├── state/              the trained model
│   ├── token.json      the tokenizer
│   ├── memory.pt       latest weights + optimizer (resume training)
│   └── peak.pt         weights with the lowest val loss (use this one)
├── translate.py        trains the tokenizer and writes the .bin files
└── run.py              console: train and chat
```
 
Scripts you run sit in `ai/`. Code that gets imported sits in `ai/model/`.
 
---
 
## Quick start
 
**1. Prepare the data** (once, and again after adding data):
 
```
cd ~/Desktop/kiwi/ai
.venv/bin/python translate.py
```
 
The first run trains the tokenizer on a weighted 400 MB sample, then encodes every `data/*.txt` into a `.bin` file. Later runs reuse the same tokenizer and only encode new or changed files.
 
**2. Start the console:**
 
```
cd ~/Desktop/kiwi/ai
.venv/bin/python run.py
```
 
| Command | What it does |
| --- | --- |
| `/train 1000` | Train 1,000 steps (no number = one interval) |
| `/save` | Save `memory.pt` now |
| `/clear` | Forget the chat history |
| `/help` | List the commands |
| `/quit` | Save and exit (Ctrl+C or Ctrl+D also work) |
 
Anything not starting with `/` is a chat message.
 
**Long runs:** keep the Mac awake and plugged in.
 
```
cd ~/Desktop/kiwi/ai
printf "/train 10000\n/quit\n" | caffeinate -is .venv/bin/python run.py
```
 
---
 
## Data naming convention
 
Every data file is named **2-letter section code + 6-digit number**:
 
```
ch000001.txt   chat, file 1
ph000003.txt   physics, file 3
```
 
- Numbers start at `000001` and count up inside each section
- A new dataset gets the next free number in its section
- Each file is one dataset (one source), so the mix can be weighted by section
- Documents inside a file are separated by one blank line
- Every file has a row in `data/sources.csv`: `file, code, dataset, MB`
- `data/naming.csv` holds the full code list below
| Priority | Code | Section | Priority | Code | Section |
| --- | --- | --- | --- | --- | --- |
| 1 | ch | chat | 14 | sc | general science |
| 2 | qa | questions and answers | 15 | cm | chemistry |
| 3 | rs | reasoning | 16 | bi | biology |
| 4 | mt | math | 17 | md | medical |
| 5 | lg | logic | 18 | ea | earth science |
| 6 | pr | programming | 19 | ec | economics |
| 7 | dc | technical docs | 20 | lw | law |
| 8 | cs | computer science | 21 | hs | history |
| 9 | en | engineering | 22 | ge | geography |
| 10 | el | electronics | 23 | pl | philosophy |
| 11 | ph | physics | 24 | ln | language |
| 12 | nu | nuclear | 25 | gn | general knowledge |
| 13 | as | astronomy | 26 | st | stories |
 
### Section weights
 
Sections are sampled by **size × weight**, so big sections don't drown out small important ones. Default weight is 1.
 
| Code | Weight | Why |
| --- | --- | --- |
| ch | 2 | Chat is the main goal but only ~3% of the raw text |
| lw | 0.5 | Case law is large and narrow |
| gn | 0.3 | General text is about half of all data |
 
```
share = size * weight / Σ(size * weight)
gn: ~50% of raw text → ~22% of batches
ch: ~3.4% of raw text → ~10% of batches
```
 
### Chat format
 
```
User: question
KiwiML: answer
```
 
---
 
## Tokenizer
 
| Property | Value |
| --- | --- |
| Type | Byte-level BPE (Hugging Face `tokenizers`) |
| Vocab | 8,192 = 256 bytes + 1 special + 7,935 merges |
| Special token | `<|eot|>` (end of text), id 0, added after every document |
| Digits | Split one per token: `2026` → `2 0 2 6` |
| Unknown characters | None. Any text encodes, since the base is all 256 bytes |
| Trained on | 400 MB sample, each section by its weighted share |
| Compression | ~3.39 chars per token on average |
| File | `state/token.json` |
 
Chars per token by kind of text:
 
| Text | Chars per token |
| --- | --- |
| English prose, chat | 3.5 to 4.3 |
| Wikipedia, books, medical | 3.1 to 3.7 |
| Code, docs | 2.5 to 3.5 |
| Math | 2.5 to 3.3 |
 
Never retrain `token.json` after training has started. A new tokenizer gives every piece a different id and the model would read nonsense.
 
---
 
## Model properties
 
Decoder-only transformer (GPT-style).
 
| Property | Value |
| --- | --- |
| Vocab | 8,192 |
| Context | 768 tokens (~2,600 characters) |
| Embed (width) | 1,024 |
| Layers | 12 |
| Heads | 16 (64 per head) |
| Feed-forward | 4,096 (4 × embed), GELU |
| Norm | LayerNorm, pre-norm |
| Positions | Learned position table |
| Output | Tied to the token embedding |
| Dropout | 0.1 |
| **Parameters** | **~160.3M** |
 
Parameter count:
 
```
one block = 12 * E² + 13 * E
12 * 1,024² + 13 * 1,024 = 12,596,224
12 blocks = 151,154,688
positions = 768 * 1,024 = 786,432
final norm = 2 * 1,024 = 2,048
vocab = 8,192 * 1,024 + 8,192 = 8,396,800
total = 160,339,968
```
 
`embed` must divide evenly by `numHeads`.
 
---
 
## Training settings
 
| Setting | Value |
| --- | --- |
| Optimizer | AdamW, betas (0.9, 0.95), weight decay 0.1 |
| Learning rate | 3e-4 peak |
| Schedule | 200-step warmup, then cosine decay to 3e-5 |
| Total steps | 130,000 |
| Batch | 32 windows × 768 tokens = 24,576 tokens per step |
| Gradient clipping | 1.0 |
| Precision | bfloat16 autocast |
| Val split | Last 1% of every file |
| Logging | Every `interval` steps: train loss, val loss, rate, seconds |
 
```
tokens trained = 130,000 * 24,576 ~ 3.2B
target = 20 * 160M = 3.2B (compute-optimal)
data = 2.97B tokens → ~1.08 passes
```
 
Expected first loss: ln(8,192) ~ 9.0.
 
To compare with the old character models, convert loss to bits per char:
 
```
bits per char = loss / (3.39 * 0.693)
```
 
---
 
## Weights and checkpoints
 
| File | Holds | Use it to |
| --- | --- | --- |
| `state/memory.pt` | Latest weights + optimizer, saved every interval | Resume training |
| `state/peak.pt` | Weights from the lowest val loss so far | Chat and share |
| `state/token.json` | The tokenizer | Needed by both |
 
Checkpoint keys:
 
| Key | Content |
| --- | --- |
| `weights` | `model.state_dict()` |
| `optimizer` | AdamW state (moments m and v) |
| `config` | The `Config` the weights were built with |
| `tokenizer` | Path of the matching `token.json` |
| `step` | Steps trained |
| `bestVal` | Lowest val loss reached |
 
Size on disk: weights + Adam m + v = 12 bytes per parameter.
 
```
160.3M * 12 ~ 1.9 GB per checkpoint
```
 
Finished models are archived in `kiwi/arc/models/` as `v1.pt`, `v2.pt` …
 
---
 
## Data sources
 
About 10 GB of text, 104 files, 2.97B tokens. Full list with sizes: `data/sources.csv`.
 
| Kind | Sources |
| --- | --- |
| Chat | Dolly, OpenAssistant 2, No Robots, DailyDialog, EmpatheticDialogues, Stack Exchange everyday sites |
| Q&A and code | Stack Overflow, Stack Exchange tech sites, Python stdlib, Rosetta Code, CodeSearchNet |
| Docs | Python docs, Rust Book, Linux kernel docs, MDN |
| Math and logic | GSM8K, AQuA-RAT, OpenWebMath, Math Stack Exchange, MathOverflow, arXiv math |
| Science | arXiv abstracts, peS2o papers, SciQ, OpenStax, Stack Exchange science sites |
| Medical | MedQuAD, PubMed abstracts, arXiv medical |
| General | FineWeb-Edu, Wikipedia, Simple Wikipedia, WikiText |
| Law | COLD Cases (Harvard LIL), Law Stack Exchange |
| Stories | TinyStories, Project Gutenberg |
 
All sources are human-written except TinyStories (`st000001`), which is short stories generated by GPT-3.5 and GPT-4. Licenses vary by source (CC0, CC BY, CC BY-SA, ODC-By, Apache 2.0, GFDL, and some non-commercial such as SciQ). Fine for personal and research use. Check each source's license before publishing the data or a model trained on it.
 
---
 
## Model history
 
| Version | Tokenizer | Params | Notes |
| --- | --- | --- | --- |
| v1 | Characters | ~3.5M | WikiText only |
| v2 | Characters | | Early character model |
| v3 | Characters (96) | 25.5M | Pretrained base, step 25,000, val 1.086 (~1.57 bits per char) |
| v4 | Characters (96) | 25.5M | v3 + chat fine-tune, step 27,000 |
| v5 | BPE (8,192) | ~160M | This version |