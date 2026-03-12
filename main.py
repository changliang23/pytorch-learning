import os
import json
import torch
import pickle

from tokenizer import CharTokenizer
from model import MiniGPT

OUTPUT_DIR = "outputs"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# flow: pretrain -> sft -> inference -> export model
def run_pretrain():
    print("=== Pretrain ===")
    text = open("data_pretrain.txt").read()
    # tokenizer from data pretrain file
    tok = CharTokenizer(text)
    # save to pkl file
    with open(f"{OUTPUT_DIR}/tokenizer.pkl", "wb") as f:
        pickle.dump(tok, f)
    # build training data(from text to data list)
    data = torch.tensor(tok.encode(text))
    # load model
    model = MiniGPT(tok.vocab_size)
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    loss_fn = torch.nn.CrossEntropyLoss()
    # create pretrained data
    for epoch in range(50):
        # guess next string
        x = data[:-1].unsqueeze(0)
        y = data[1:].unsqueeze(0)
        logits = model(x)
        loss = loss_fn(logits.view(-1, tok.vocab_size), y.view(-1))
        opt.zero_grad()
        loss.backward()
        opt.step()

        if epoch % 10 == 0:
            print("epoch:", epoch, "loss:", loss.item())
    # save pretrained pt
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/pretrained.pt")

    return tok


def run_sft():
    print("=== SFT ===")
    # load sft data
    data = json.load(open("data_sft.json"))
    # load tokenizer pkl
    with open(f"{OUTPUT_DIR}/tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    # load pretrained pt
    model = MiniGPT(tok.vocab_size)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/pretrained.pt"))
    # optimizer
    opt = torch.optim.Adam(model.parameters(), lr=1e-4)
    loss_fn = torch.nn.CrossEntropyLoss()
    # ctreate sft pt
    for epoch in range(50):
        for d in data:
            # training data
            s = d["instruction"] + d["output"]
            ids = torch.tensor(tok.encode(s)).unsqueeze(0)
            # guess next token
            x = ids[:, :-1]
            y = ids[:, 1:]
            logits = model(x)
            loss = loss_fn(logits.view(-1, tok.vocab_size), y.view(-1))
            opt.zero_grad()
            loss.backward()
            opt.step()

        if epoch % 10 == 0:
            print("epoch:", epoch, "loss:", loss.item())
    # save sft pt
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/sft.pt")

    return tok

# input prompt return generated string
def run_infer(prompt="hello"):
    print("=== Inference ===")
    # load tokenizer pkl
    with open(f"{OUTPUT_DIR}/tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    # load sft pt
    model = MiniGPT(tok.vocab_size)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/sft.pt"))
    model.eval()
    ids = tok.encode(prompt)
    # guess next token
    for _ in range(20):
        x = torch.tensor(ids).unsqueeze(0)
        logits = model(x)
        next_id = torch.argmax(logits[0, -1]).item()
        ids.append(next_id)
    # decode token (to string)
    result = tok.decode(ids)

    print("prompt:", prompt)
    print("generated:", result)

    return result


def export_model():
    print("=== Export Model ===")
    # load tokenizer pkl
    with open(f"{OUTPUT_DIR}/tokenizer.pkl", "rb") as f:
        tok = pickle.load(f)
    # load sft pt
    model = MiniGPT(tok.vocab_size)
    model.load_state_dict(torch.load(f"{OUTPUT_DIR}/sft.pt"))
    # save to final pt
    torch.save(model.state_dict(), f"{OUTPUT_DIR}/final.pt")
    # generate random int tensor
    dummy = torch.randint(0, tok.vocab_size, (1, 10))
    # save to onnx
    torch.onnx.export(
        model,
        dummy,
        f"{OUTPUT_DIR}/model.onnx",
        input_names=["input"],
        output_names=["output"]
    )
    print("model exported")
    return f"{OUTPUT_DIR}/final.pt"


def main():

    tok = run_pretrain()
    tok = run_sft()
    text = run_infer("hello")
    final_model = export_model()

    print("\n=== Pipeline Finished ===")
    print("Generated text:", text)
    print("Final model:", final_model)

    return final_model


if __name__ == "__main__":
    main()