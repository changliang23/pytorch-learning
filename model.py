import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, d_model)
        self.lm_head = nn.Linear(d_model, vocab_size)

    def forward(self, x):
        h = self.embed(x)
        logits = self.lm_head(h)
        return logits
