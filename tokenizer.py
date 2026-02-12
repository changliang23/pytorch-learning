import json


class CharTokenizer:
    def __init__(self, text):
        chars = sorted(list(set(text)))
        self.vocab = ["<unk>"] + chars
        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        self.itos = {i:ch for i,ch in enumerate(self.vocab)}
        self.vocab_size = len(self.vocab)

    def encode(self, s):
        return [self.stoi.get(c, self.stoi["<unk>"]) for c in s]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])

