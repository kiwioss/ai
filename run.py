from model.config import Config, Train
from model.network import Network
from model.tokens import encode, decode

class Run:
    def __init__(self, config: Config, train: Train):
        self.config = config
        self.train = train
        self.model = Network(config.context, config.embed, config.numHeads, config.numLayers, config.dropOut)

    def console(self):
        while True:
            text = input("> ")
            ids = encode(text, self.model.toId)
            output_ids = self.model.query(ids)
            output_text = decode(output_ids, self.model.toChar)
            print(output_text)