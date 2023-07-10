from embedding_models.vits8.model import ViTStamp
import torch
from PIL import Image

def test_vits8():
    img_cropped = Image.open("tests/test_data/test_cropped_stamp.png")
    model = ViTStamp()
    output = model(image=img_cropped)
    assert output.shape == torch.Size([1,384]), False
