import json, torch, pickle
from model import MiniGPT

# flow: sft file -> load tokenizer -> load model -> instruction + output -> token ids -> miniGPT - > cross entropy loss -> optim -> save sft model
def run_sft():
    data = json.load(open("data_sft.json"))

    # load tokenizer
    with open("tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)

    # load pretrain data
    model = MiniGPT(tok.vocab_size)
    model.load_state_dict(torch.load("pretrained.pt"))

    # optim, para 1e-4 (less than pretrain, bcz this is fine-tuning)
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()

    for epoch in range(100):
        for d in data:
            s = d["instruction"] + d["output"]
            # flow: text -> encode -> token ids -> tensor
            ids = torch.tensor(tok.encode(s)).unsqueeze(0)
            # construct training data for model
            x, y = ids[:, :-1], ids[:, 1:]
            # calc forward
            logits = model(x)
            # calc loss
            loss = loss_fn(logits.view(-1, tok.vocab_size), y.view(-1))
            # backpropagation -> update model
            opt.zero_grad()
            loss.backward()
            opt.step()
    # save sft model
    torch.save(model.state_dict(), "sft.pt")
    return tok
