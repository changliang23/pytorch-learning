import torch
from model import MiniGPT

# flow: load model -> save final model -> dummp input -> export onnx
model = MiniGPT(100)
# save final model
torch.save(model.state_dict(), "final.pt")

# shape (1,10), vocab_size 100
dummy = torch.randint(0,100,(1,10))
# output to onnx
torch.onnx.export(model, dummy, "model.onnx", input_names=["input"],
    output_names=["output"])

