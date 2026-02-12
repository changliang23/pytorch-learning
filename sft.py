import json, torch, pickle
from model import MiniGPT

def run_sft():
    data = json.load(open("data_sft.json"))

    # ✅ 加载同一个 tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    model = MiniGPT(tok.vocab_size)
    model.load_state_dict(torch.load("pretrained.pt"))

    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        for d in data:
            s = d["instruction"] + d["output"]
            ids = torch.tensor(tok.encode(s)).unsqueeze(0)
            x, y = ids[:, :-1], ids[:, 1:]
            logits = model(x)
            loss = loss_fn(logits.view(-1, tok.vocab_size), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

    torch.save(model.state_dict(), "sft.pt")
    return tok
