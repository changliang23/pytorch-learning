import torch
from model import MiniGPT

policy = MiniGPT(100)
reward_model = MiniGPT(100)

def reward_fn(text):
    return torch.tensor(1.0 if "好" in text else 0.0)

# 伪代码示意
for step in range(10):
    output = policy(torch.randint(0,100,(1,10)))
    reward = reward_fn("好回答")
    loss = -reward * output.mean()
    loss.backward()
