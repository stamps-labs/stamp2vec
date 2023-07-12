from typing import Any
from ultralytics import YOLO
from huggingface_hub import hf_hub_download
import torch


class Yolov8Pipeline:
    def __init__(self):
        self.model = None

    @classmethod
    def from_pretrained(cls, model_path_hf: str = None, filename_hf: str = None, local_model_path: str = None):
        yolo = cls()
        if model_path_hf is not None and filename_hf is not None:
            yolo.model = YOLO(hf_hub_download(model_path_hf, filename=filename_hf), task='detect')
        elif local_model_path is not None:
            yolo.model = YOLO(local_model_path)
        return yolo

    def __call__(self, image) -> torch.Tensor:
        shape = torch.tensor(image.size)
        image = image.convert('RGB')
        coef = torch.hstack((shape, shape)) / 640
        image = image.resize((640, 640))
        boxes = self.model(image, verbose=False)[0].boxes.xyxy.cpu()
        boxes = boxes * coef
        return boxes