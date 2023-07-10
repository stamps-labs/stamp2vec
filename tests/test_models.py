from embedding_models.vits8.model import ViTStamp
from embedding_models.vae.model import Encoder
from detection_model.model import YOLOStamp
from ultralytics import YOLO
from torchvision.transforms.functional import to_tensor
from huggingface_hub import hf_hub_download
import torch
import numpy as np
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image
import pytest
from detection_model.utils import *

class TestModels:

    @pytest.fixture
    def config(self):
        config = {}
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        config["img_classify"] = Image.open("tests/test_data/test_cropped_stamp.png")
        config["img_detect"] = Image.open("tests/test_data/test_full_doc.jpeg")
        return config

    def test_vits8(self, config):
        model = ViTStamp()
        output = model(image=config["img_classify"])
        assert output.shape == torch.Size([1,384]), False

    def test_vae(self, config):
        model = Encoder()
        model.load_state_dict(torch.load(hf_hub_download("stamps-labs/vae-encoder", filename="encoder.pth"), map_location="cpu"))
        model.to(device=config["device"])
        model.eval()
        img_tensor = to_tensor(config["img_classify"].resize((118,118)))
        output = model(img_tensor.unsqueeze(0).to(config["device"]))[0][0].detach().cpu()
        assert output.shape == torch.Size([128])
    
    def test_yolov8(self, config):
        model = YOLO(hf_hub_download("stamps-labs/yolov8-finetuned", filename="best.torchscript"), task="detect")
        image = config["img_detect"].resize((640, 640))
        boxes = model(image)[0].boxes.xyxy.cpu()
        assert boxes.shape == torch.Size([1, 4])
    
    def test_yolostamp(self, config):
        model = YOLOStamp()
        model.load_state_dict(torch.load(hf_hub_download('stamps-labs/yolo-stamp', filename='state_dict.pth'), map_location='cpu'))
        model.to(config['device'])
        transform = A.Compose([
        A.Normalize(),
        ToTensorV2(p=1.0),
        ])
        image_tensor = transform(image=np.array(config["img_detect"]))['image']
        output = model(image_tensor.unsqueeze(0).to(config["device"]))
        assert output.shape == torch.Size([1,3,3,3,5])