import json


class CharTokenizer:
    def __init__(self, text):
        # get chars from text file
        chars = sorted(list(set(text)))
        # build vocabulary
        self.vocab = ["<unk>"] + chars
        # string to index
        self.stoi = {ch:i for i,ch in enumerate(self.vocab)}
        # index to string
        self.itos = {i:ch for i,ch in enumerate(self.vocab)}
        # calc vocabulary size
        self.vocab_size = len(self.vocab)

    def encode(self, s):
        return [self.stoi.get(c, self.stoi["<unk>"]) for c in s]

    def decode(self, ids):
        return "".join([self.itos[i] for i in ids])

# text = 'Test Demo'
# tokenizer = CharTokenizer(text)
#
# print("==== Tokenizer Info ====")
# print("vocab_size:", tokenizer.vocab_size)
# print("vocab:", tokenizer.vocab)
#
# print("\n==== Encode Test ====")
# encoded = tokenizer.encode(tokenizer.stoi)
# print("encoded:", encoded)
#
# print("\n==== Decode Test ====")
# print("decoded:", tokenizer.decode(encoded))
