import torch
import model

torch.save(model.state_dict(), "final.pt")

dummy = torch.randint(0,100,(1,10))
torch.onnx.export(model, dummy, "model.onnx")

