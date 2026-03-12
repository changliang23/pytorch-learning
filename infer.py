import torch, pickle
from model import MiniGPT

# flow: input prompt -> tokenizer -> load sft model -> create token -> decode to text
def run_infer(prompt="hello"):
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    # load sft model
    model = MiniGPT(tok.vocab_size)
    model.load_state_dict(torch.load("sft.pt"))
    model.eval()

    def generate(prompt, max_len=20):
        # base on prompt to create token
        ids = tok.encode(prompt)
        for _ in range(max_len):
            x = torch.tensor(ids).unsqueeze(0)
            # calc forward
            logits = model(x)
            # choose next token
            next_id = torch.argmax(logits[0, -1]).item()
            # add token to list
            ids.append(next_id)
        return tok.decode(ids)

    return generate(prompt)
