import torch
from pipelines.feature_extraction.vae import VaePipeline
from pipelines.feature_extraction.vits8 import Vits8Pipeline
from pipelines.detection.yolo_stamp import YoloStampPipeline
from pipelines.detection.yolo_v8 import Yolov8Pipeline
from PIL import Image
import pytest

class TestModels:

    @pytest.fixture
    def config(self):
        config = {}
        config["device"] = "cuda" if torch.cuda.is_available() else "cpu"
        config["img_classify"] = Image.open("tests/test_data/test_cropped_stamp.png")
        config["img_detect"] = Image.open("tests/test_data/test_full_doc.jpeg")
        return config

    def test_vits8(self, config):
        pipe = Vits8Pipeline.from_pretrained("stamps-labs/vits8-stamp", filename_hf="vits8stamp-torchscript.pth")
        output = pipe(image=config["img_classify"])
        assert output.shape == (384, )

    def test_vae(self, config):
        pipe = VaePipeline.from_pretrained("stamps-labs/vae-encoder", filename_hf="encoder.pth")
        output = pipe(image=config["img_classify"])
        assert output.shape == torch.Size([128])
    
    def test_yolov8(self, config):
        pipe = Yolov8Pipeline.from_pretrained('stamps-labs/yolov8-finetuned', filename_hf="best.torchscript")
        boxes = pipe(image=config["img_detect"])
        assert boxes.shape == torch.Size([1, 4])
    
    def test_yolostamp(self, config):
        pipe = YoloStampPipeline.from_pretrained('stamps-labs/yolo-stamp', filename_hf='state_dict.pth')
        output = pipe(image=config["img_detect"])
        assert output.shape == torch.Size([1, 4])