import torch, json
from tokenizer import CharTokenizer
import torch.nn.functional as F
from model import MiniGPT

# flow: input prompt -> policy create token -> decode text -> reward_fn -> update policy
text = open("data_pretrain.txt").read()
tok = CharTokenizer(text)
policy = MiniGPT(tok.vocab_size)
# optim, para 0.001
optimizer = torch.optim.Adam(policy.parameters(), lr=1e-3)


def reward_fn(text):
    if "hello" in text:
        return torch.tensor(1.0)
    return torch.tensor(0.0)

def decode(ids):
    return "".join([chr(65 + int(i % 26)) for i in ids])


for step in range(10):
    # input prompt,rand token
    prompt = torch.randint(0, tok.vocab_size, (1, 10))
    # forward
    logits = policy(prompt)
    # get latest token to calc prob
    probs = F.softmax(logits[:, -1, :], dim=-1)
    token = torch.multinomial(probs, 1)
    # create file
    text = decode(token[0])
    reward = reward_fn(text)
    # calc reward update loss
    loss = -reward * torch.log(probs[0, token])
    # update
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    print("step:", step)
    print("generated:", text)
    print("reward:", reward.item())