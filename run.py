#Import--------------------------------------------------------------------------------
try:
    import readline
except ImportError:
    pass

from dataclasses import asdict
from pathlib import Path

import torch

from model.config import Config, Train
from model.network import Machine
from model.tokens import encode, decode, build, filter


#Paths--------------------------------------------------------------------------------
root = Path(__file__).resolve().parent #The ai/ folder from any working directory
dataDir = root / "data"
weightsDir = root / "weight" / "model.pt" #The checkpoint file


#Helpers--------------------------------------------------------------------------------
def optimalDevice():
    if torch.cuda.is_available():
        return torch.device("cuda") #NVIDIA GPU
    if torch.backends.mps.is_available():
        return torch.device("mps") #Apple Silicon GPU
    return torch.device("cpu")

def readData(filePath):
    files = sorted(filePath.glob("*.txt")) #Sorted so the text is identical every run
    return "\n".join([filter(file.read_text(encoding="utf-8")) for file in files])


#Main--------------------------------------------------------------------------------
class Run:
    def __init__(self, config: Config, practice: Train):
        self.device = optimalDevice()
        self.temperature = 0.5 #Reply randomness
        self.config = config
        self.practice = practice
        self.length = 1000 #Max characters per reply
        self.history = "" #The conversation so far
        self.step = 0 #Total training steps
        self.running = True #False ends the loop
        self.trainData = None #Encoded data, cached after the first /train
        self.valData = None

        saved = torch.load(weightsDir, map_location=self.device) if weightsDir.exists() else None
        if saved:
            self.config = Config(**saved["config"]) #Checkpoint shape so the weights fit
            self.vocab = saved["vocab"] #Character IDs the weights were trained on
        else:
            self.vocab = ["\n"] + [chr(i) for i in range(32, 127)] #Newline with new 95 plain chars
            if not self.vocab:
                raise SystemExit("no data found")
        self.toId, self.toChar = build(self.vocab)

        c = self.config
        self.model = Machine(numChars=len(self.vocab), context=c.context, embed=c.embed, numHeads=c.numHeads, numLayers=c.numLayers, dropOut=c.dropOut).to(self.device) #Named arguments prevent order mistakes
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr = practice.rate)

        if saved:
            self.model.load_state_dict(saved["weights"]) #Replace the random weights
            self.optimizer.load_state_dict(saved["optimizer"])
            self.step = saved["step"]

        self.model.eval() #Dropout off until training

        self.commands = { #Maps each name to its method and description
            "/help":  (self.showHelp, "display commands"),
            "/train": (self.trainModel, "train steps"),
            "/clear": (self.clear, "clear history"),
            "/save":  (self.saveModel, "save checkpoint"),
            "/quit":  (self.quit, "exit console"),
        }

    def loop(self):
        params = sum(p.numel() for p in self.model.parameters()) #Tied weights count once
        print(f"{params:,} parameters | device {self.device} | step {self.step} | /help for commands")
        while self.running:
            try:
                line = input("User: ")
            except (KeyboardInterrupt, EOFError): #Ctrl+C or Ctrl+D at the prompt
                print()
                self.quit([])
                break
            self.inputProcess(line)

    def inputProcess(self, line):
        line = line.strip()
        if not line:
            return
        if line.startswith("/"):
            name, *args = line.split() #"/train 500" becomes "/train" and ["500"]
            if name in self.commands:
                func, _ = self.commands[name]
                try:
                    func(args)
                except Exception as e: #Errors print instead of ending the session
                    print(e)
            else:
                print("Unknown command")
        else:
            first = line.split()[0]
            if "/" + first in self.commands: #Catches a forgotten slash
                print(f"did you mean /{first}?")
                return
            try:
                self.queryChar(line)
            except Exception as e:
                print(e)

    def queryChar(self, line):
        line = filter(line)
        self.history += f"User: {line}\nKiwiML: " #Labels must match the training data
        prompt = self.history[-self.config.context:] #Only what fits in the context window
        ids = torch.tensor([encode(prompt, self.toId)], dtype=torch.long, device=self.device) #A batch of one
        if ids.shape[1] == 0:
            print("vocab incapability")
            return

        end = self.toId.get("\n") #Replies stop at a newline
        output = self.model.query(ids, self.length, self.temperature, end)
        generation = decode(output[0, ids.shape[1]:].tolist(), self.toChar) #New characters only

        print("KiwiML: " + generation.rstrip("\n"))
        self.history += generation if generation.endswith("\n") else generation + "\n"

    def trainModel(self, args):
        count = int(args[0]) if args else self.practice.interval #Steps from "/train 500" or one interval
        if self.trainData is None: #Only the first /train reads and encodes the files
            text = readData(dataDir)
            unknown = set(text) - set(self.vocab)
            if unknown:
                print(f"unknown char : {sorted(unknown)[:10]}")
                return
            data = torch.tensor(encode(text, self.toId), dtype=torch.long)
            split = int(0.9 * len(data))
            self.trainData, self.valData = data[:split], data[split:]
        trainData, valData = self.trainData, self.valData

        self.model.train() #Dropout on

        try:
            for _ in range(count):
                x, y = self.getBatch(trainData)
                with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16): #Faster 16-bit forward pass
                    _, loss = self.model(x, y) #Forward pass and loss
                self.optimizer.zero_grad() #Clear the previous gradients
                loss.backward() #Backpropagation
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step() #Update the weights
                self.step += 1
                if self.step % self.practice.interval == 0:
                    print(f"step {self.step} | train {self.lossEstimate(trainData):.3f}" f" | val {self.lossEstimate(valData):.3f}")
        except KeyboardInterrupt: #Ctrl+C stops training but keeps the console
            print("\nkeyboard exception")
        self.model.eval() #Dropout off again
        self.saveModel([])

    def optimizedTraining(self,args): #Testing Schedule
        count = int(args[0]) if args else self.practice.interval
        count /= 10
        count = int(count)
        for r in range(11):
            self.trainModel(count*r)
            print(f"step {self.step}")

    def getBatch(self, source):
        c = self.config.context
        starts = torch.randint(len(source) - c - 1, (self.practice.batch,)) #Random start positions
        x = torch.stack([source[s:s + c] for s in starts]) #Inputs
        y = torch.stack([source[s + 1:s + c + 1] for s in starts]) #Inputs shifted one character right
        return x.to(self.device), y.to(self.device)

    @torch.no_grad() #Measure without learning
    def lossEstimate(self, source):
        self.model.eval()
        with torch.autocast(device_type=self.device.type, dtype=torch.bfloat16):
            losses = [self.model(*self.getBatch(source))[1].item() for _ in range(20)]
        self.model.train()
        return sum(losses) / len(losses)

    def saveModel(self, args):
        weightsDir.parent.mkdir(exist_ok=True) #Create the folder if missing
        torch.save({
            "weights": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(), #Lets training resume smoothly
            "vocab": self.vocab,
            "config": asdict(self.config), #The shape these weights fit
            "step": self.step,
        }, weightsDir)
        print(f"saved at step {self.step}")

    def clear(self, args):
        self.history = ""

    def showHelp(self, args):
        for name, (_, description) in self.commands.items():
            print(f"  {name:<10}{description}") #Pads names so descriptions line up

    def quit(self, args):
        self.saveModel([])
        self.running = False


#Run--------------------------------------------------------------------------------
if __name__ == "__main__": #Runs only when started directly
    Run(Config(), Train()).loop()