import torch
from model_oml import EmbeddingModelOML
from PIL import Image



if __name__ == "__main__":
    model = EmbeddingModelOML().extractor.cuda() #Maybe need to be changed to be flexible, perhaps just EmbeddingModelOML().extractor

    model.eval()

    with torch.no_grad():
        model_ts = torch.jit.script(model)

    model_ts.save("../models/vits8stamp-torchscript.pth")