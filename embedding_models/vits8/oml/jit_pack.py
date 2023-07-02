import torch
from model_oml import EmbeddingModel
from PIL import Image



if __name__ == "__main__":
    model = EmbeddingModel().extractor.cuda()

    model.eval()

    with torch.no_grad():
        model_ts = torch.jit.script(model)

    model_ts.save("../models/vits8stamp-torchscript.pth")