from typing import Any
from detection_models.yolo_stamp.model import YOLOStamp
from detection_models.yolo_stamp.constants import *
from detection_models.yolo_stamp.utils import *
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
import torch
from huggingface_hub import hf_hub_download
import numpy as np

class YoloStampPipeline:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = YOLOStamp()
        self.transform = A.Compose([
            A.Normalize(),
            ToTensorV2(p=1.0),
        ])
    
    @classmethod
    def from_pretrained(cls, model_path_hf: str = None, filename_hf: str = None, local_model_path: str = None):
        yolo = cls()
        if model_path_hf is not None and filename_hf is not None:
            yolo.model.load_state_dict(torch.load(hf_hub_download(model_path_hf, filename=filename_hf), map_location="cpu"))
            yolo.model.to(yolo.device)
            yolo.model.eval()
        elif local_model_path is not None:
            yolo.model.load_state_dict(torch.load(local_model_path, map_location="cpu"))
            yolo.model.to(yolo.device)
            yolo.model.eval()
        return yolo
    
    def __call__(self, image) -> torch.Tensor:
        shape = torch.tensor(image.size)
        coef =  torch.hstack((shape, shape)) / 448
        image = image.convert("RGB").resize((448, 448))
        image_tensor = self.transform(image=np.array(image))["image"]
        output = self.model(image_tensor.unsqueeze(0).to(self.device))
        boxes = output_tensor_to_boxes(output[0].detach().cpu())
        boxes = nonmax_suppression(boxes=boxes)
        boxes = xywh2xyxy(torch.tensor(boxes)[:, :4])
        boxes = boxes * coef
        return boxes