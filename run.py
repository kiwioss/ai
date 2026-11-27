#Import--------------------------------------------------------------------------------
try:
    import readline #arrow keys and history at the input prompt on macos and linux
except ImportError:
    pass

from tokenizers import Tokenizer
from dataclasses import asdict
from pathlib import Path
from tqdm import tqdm

import torch
import time
import numpy as np
import math
import random

from model.config import Config, Train
from model.network import Machine


#Paths--------------------------------------------------------------------------------
root = Path(__file__).resolve().parent #ai folder from any working directory
dataDir = root / "data" #txt sources and the bin token files made by prep.py
stateDir = root / "state" #tokenizer and saved weights
memoryDir = stateDir / "memory.pt" #latest weights and optimizer used to resume training
recordDir = stateDir / "record.pt" #weights from the lowest val loss so far use this one to chat or share
vocabDir = stateDir / "token.json" #bpe tokenizer made once by prep.py


#Settings--------------------------------------------------------------------------------
sectionWeights = {"gn": 0.3, "lw": 0.5, "ch": 2} #how often each section is sampled as size times weight and any code not listed is 1


#Helpers--------------------------------------------------------------------------------
def optimalDevice():
    if torch.cuda.is_available():
        return torch.device("cuda") #nvidia gpu
    if torch.backends.mps.is_available():
        return torch.device("mps") #apple silicon gpu
    return torch.device("cpu") #fallback and very slow


#Main--------------------------------------------------------------------------------
class Run:
    def __init__(self, config: Config, practice: Train):
        self.device = optimalDevice()
        self.temperature = 0.75 #reply randomness where lower is safer and higher is more varied
        self.config = config #model shape replaced by the checkpoint config if one exists
        self.practice = practice #training settings like batch accumulate rate interval and ceiling
        self.length = 500 #max tokens per reply
        self.history = "" #conversation so far
        self.step = 0 #total training steps
        self.running = True #false ends the loop
        self.memmap = None #section code to list of train and val pairs filled by the first /train
        self.recordVal = float("inf") #lowest val loss so far starts at infinity so the first check always wins

        #tokenizer must exist before the model because it sets the vocab
        if vocabDir.exists():
            self.tokenizer = Tokenizer.from_file(str(vocabDir)) #from_file needs a string not a path
        else:
            raise SystemExit("inference not found")

        #checkpoint config wins so the saved weights fit the model built below
        saved = torch.load(memoryDir, map_location=self.device) if memoryDir.exists() else None

        if saved:
            self.config = Config(**saved["config"])

        if self.tokenizer.get_vocab_size() != self.config.vocab: #checked after loading the config so it compares with what the weights were trained on
            raise SystemExit("tokenizer and config vocab differ")

        self.eot = self.tokenizer.token_to_id("<|eot|>") #end of text id where replies stop

        #model and optimizer
        c = self.config
        self.model = Machine(vocab = c.vocab, context = c.context, embed = c.embed, numHeads = c.numHeads, numLayers = c.numLayers, dropOut = c.dropOut).to(self.device)

        #adamw where lr is the step size and betas set how much of the recent gradients it remembers with 0.95 as short memory so spikes fade fast
        #weight decay pulls every weight a little toward 0 each step by lr times 0.1 so weights stay small and do not memorize
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = practice.rate, betas = (0.9, 0.95), weight_decay = 0.1)

        if saved:
            self.model.load_state_dict(saved["weights"])
            self.optimizer.load_state_dict(saved["optimizer"]) #restores adam averages so resuming does not cause a loss bump
            self.step = saved["step"]
            self.recordVal = saved.get("recordVal", float("inf")) #record survives restarts

        for m in self.optimizer.param_groups:
            m["betas"] = (0.9, 0.95) #only matters for old checkpoints saved before betas were set in adamw above and safe to delete later

        self.model.eval() #dropout off until training

        self.commands = { #command name to method and description
            "/help":  (self.showHelp, "display commands"),
            "/train": (self.trainModel, "train steps"),
            "/clear": (self.clear, "clear history"),
            "/save":  (self.saveModel, "save checkpoint"),
            "/quit":  (self.quit, "exit console"),
            "/loss":  (self.checkLoss, "display training loss"),
        }


    #Console--------------------------------------------------------------------------------
    def loop(self):
        params = sum(p.numel() for p in self.model.parameters()) #tied weights count once
        print(f"{params:,} parameters | device {self.device} | step {self.step} | /help for commands")

        while self.running:
            try:
                line = input("User: ")
            except (KeyboardInterrupt, EOFError):
                print()
                self.quit([])
                break

            self.inputProcess(line)

    def inputProcess(self, line):
        line = line.strip()
        if not line:
            return

        if line.startswith("/"):
            name, *args = line.split() #/train 500 becomes /train and 500

            if name in self.commands:
                func, _ = self.commands[name]
                try:
                    func(args)
                except Exception as e: #errors print instead of ending the session
                    print(e)
            else:
                print("Unknown command")
        else:
            first = line.split()[0]
            if "/" + first in self.commands: #catches a forgotten slash like train 500
                print(f"use /{first}")
                return

            try:
                self.query(line)
            except Exception as e:
                print(e)


    #Chat--------------------------------------------------------------------------------
    def query(self, line):
        self.history += f"User: {line}\nKiwiML: " #labels must match the training data

        ids = self.tokenizer.encode(self.history).ids[-self.config.context:] #last tokens that fit the context window
        ids = torch.tensor([ids], dtype=torch.long, device=self.device) #a batch of one

        output = self.model.query(ids, self.length, self.temperature, self.eot)

        generation = self.tokenizer.decode(output[0, ids.shape[1]:].tolist()) #new tokens only
        generation = generation.split("\nUser:")[0].strip() #cut if it starts writing your next turn

        print("KiwiML: " + generation)
        self.history += generation + "\n"


    #Training--------------------------------------------------------------------------------
    def trainModel(self, args):
        #print(self.practice)shows the real batch accumulate rate interval and ceiling

        count = int(args[0]) if args else self.practice.interval #/train 500 runs 500 steps and /train alone runs one interval

        if self.memmap is None: #only the first /train opens the bin files
            self.loadData()

        self.model.train() #dropout on

        try:
            trainingInterval = time.time()

            for _ in tqdm(range(count)):
                self.optimizer.zero_grad() #clear once per real step

                for _ in range(self.practice.accumulate): #small batches added up into one real batch
                    x, y = self.getBatch("train")
                    with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16): #16 bit math for speed
                        _, loss = self.model(x, y) #forward pass and loss
                    (loss / self.practice.accumulate).backward() #divide so the added gradients become an average

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0) #cap the step size so one bad batch cannot cause a spike

                for prm in self.optimizer.param_groups:
                    prm["lr"] = self.rateAt(self.step) #warmup then cosine decay

                self.optimizer.step() #update the weights
                self.step += 1

                #every interval measure then keep the best version then save
                if self.step % self.practice.interval == 0:
                    train, val = self.lossEstimate("train"), self.lossEstimate("val")
                    rate = self.optimizer.param_groups[0]["lr"]
                    tqdm.write(f"step {self.step} | train {train:.3f} | val {val:.3f} | lr {rate:.1e} | seconds {int(time.time() - trainingInterval)}")

                    if val < self.recordVal: #new record so keep this version
                        self.recordVal = val
                        self.saveModel([], recordDir)

                    self.saveModel([]) #every interval so a crash or dead battery loses one interval at most
                    trainingInterval = time.time()

        except KeyboardInterrupt:
            print("\nkeyboard exception")

        self.model.eval() #dropout off
        self.saveModel([])

    def loadData(self):
        self.memmap = {} #section code to list of train and val pairs with one pair per bin file

        for b in sorted(dataDir.glob("*.bin")):
            data = np.memmap(b, dtype=np.uint16, mode="r") #reads from disk on demand with almost no ram
            split = int(0.99 * len(data)) #last 1 percent of every file is val
            self.memmap.setdefault(b.stem[:2], []).append((data[:split], data[split:])) #b.stem[:2] is the section code like ch

        if not self.memmap:
            raise SystemExit(".bin files not found")

        #computed once and used by getBatch on every step
        self.codes = list(self.memmap) #all section codes
        self.codeWeights = [sectionWeights.get(k, 1) * sum(len(t) for t, _ in self.memmap[k]) for k in self.codes] #section size in train tokens times its weight

    def rateAt(self, step):
        warm, top = 200, self.practice.rate

        if step < warm:
            return top * (step + 1) / warm #warmup from 0 to top over 200 steps

        progress = min(1, (step - warm) / max(1, self.practice.ceiling - warm))
        low = top / 10
        return low + 0.5 * (top - low) * (1 + math.cos(math.pi * progress)) #cosine decay from top down to top over 10

    def getBatch(self, source):
        c = self.config.context
        content, answer = [], [] #content is what the model reads and answer is the right next token at every spot

        for _ in range(self.practice.batch):
            code = random.choices(self.codes, self.codeWeights)[0] #pick a section by weight

            parts = [t if source == "train" else v for t, v in self.memmap[code]]
            parts = [d for d in parts if len(d) > c + 1] #skip pieces shorter than one window like tiny val parts

            data = random.choices(parts, [len(d) for d in parts])[0] #pick a file with bigger files picked more often
            s = random.randrange(len(data) - c - 1) #pick a random start

            chunk = torch.from_numpy(data[s:s + c + 1].astype(np.int64)) #c plus 1 tokens turned from uint16 into int64 for the embedding
            content.append(chunk[:-1]) #what the model reads
            answer.append(chunk[1:]) #same tokens shifted by 1 so each position has the right next token

        return torch.stack(content).to(self.device), torch.stack(answer).to(self.device) #both shaped batch by context

    @torch.no_grad() #measuring only with no gradients
    def lossEstimate(self, source):
        self.model.eval()

        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            losses = [self.model(*self.getBatch(source))[1].item() for _ in range(20)] #average of 20 random batches

        self.model.train()
        return sum(losses) / len(losses)


    #Saving--------------------------------------------------------------------------------
    def saveModel(self, args, path = memoryDir): #memory.pt by default or record.pt when a new record is set
        path.parent.mkdir(exist_ok=True)

        torch.save({
            "weights": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), #lets training resume smoothly
            "tokenizer": str(vocabDir), #which tokenizer these weights need
            "recordVal": self.recordVal, #lowest val loss so it survives restarts
            "config": asdict(self.config), #model shape these weights fit
            "step": self.step,
        }, path)

        print(f"saved {path} at step {self.step}")


    #Commands--------------------------------------------------------------------------------
    def clear(self, args):
        self.history = ""

    def showHelp(self, args):
        for name, (_, description) in self.commands.items():
            print(f"  {name:<10}{description}")

    def quit(self, args):
        self.saveModel([])
        self.running = False

    def checkLoss(self, args):
        if self.memmap is None: #open the bin files if no /train has run yet
            self.loadData()
        train, val = self.lossEstimate("train"), self.lossEstimate("val")
        self.model.eval() #lossEstimate turns training mode back on at the end
        print(f"step {self.step} | train {train:.3f} | val {val:.3f}")


#Run--------------------------------------------------------------------------------
if __name__ == "__main__": #runs only when started directly and not when imported
    Run(Config(), Train()).loop()