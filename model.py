import torch
import torch.nn as nn

class MiniGPT(nn.Module):
    def __init__(self, vocab_size, d_model=64):
        super().__init__()
        # token ids to vector
        self.embed = nn.Embedding(vocab_size, d_model)
        # hidden vector to predict token probability
        self.lm_head = nn.Linear(d_model, vocab_size)

    # input token return logit
    def forward(self, x):
        # flow: token ids-> embedding -> token vector
        h = self.embed(x)
        # flow: hidden vector -> linear -> token logit
        logits = self.lm_head(h)
        return logits
