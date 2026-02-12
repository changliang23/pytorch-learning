import torch, pickle
from tokenizer import CharTokenizer
from model import MiniGPT
import json

text1 = open("data_pretrain.txt").read()
sft = json.load(open("data_sft.json"))
text2 = "".join([d["instruction"] + d["output"] for d in sft])

tok = CharTokenizer(text1 + text2)

def run_pretrain():
    text = open("data_pretrain.txt").read()
    tok = CharTokenizer(text)

    # 保存 tokenizer
    with open("tokenizer.pkl", "wb") as f:
        pickle.dump(tok, f)

    data = torch.tensor(tok.encode(text))

    model = MiniGPT(tok.vocab_size)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(200):
        x = data[:-1].unsqueeze(0)
        y = data[1:].unsqueeze(0)
        logits = model(x)
        loss = loss_fn(logits.view(-1, tok.vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

    torch.save(model.state_dict(), "pretrained.pt")
    return tok
