import torch, pickle
from model import MiniGPT

def run_infer(prompt="你好"):
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    model = MiniGPT(tok.vocab_size)
    model.load_state_dict(torch.load("sft.pt"))

    def generate(prompt, max_len=20):
        ids = tok.encode(prompt)
        for _ in range(max_len):
            x = torch.tensor(ids).unsqueeze(0)
            logits = model(x)
            next_id = torch.argmax(logits[0, -1])
            ids.append(next_id.item())
        return tok.decode(ids)

    return generate(prompt)
