from typing import Any
from huggingface_hub import hf_hub_download
import torchvision
from torchvision.transforms import ToTensor
import torch

class Yolov8Pipeline:
    def __init__(self):
        self.model = None
        self.transform = ToTensor()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    @classmethod
    def from_pretrained(cls, model_path_hf: str = None, filename_hf: str = "weights.pt", local_model_path: str = None):
        yolo = cls()
        if model_path_hf is not None and filename_hf is not None:
            yolo.model = torch.jit.load(hf_hub_download(model_path_hf, filename=filename_hf), map_location='cpu')
        elif local_model_path is not None:
            yolo.model = torch.jit.load(local_model_path)
        return yolo
    
    def __call__(self, image, nms_threshold: float = 0.45, conf_threshold: float = 0.15):
        shape = torch.tensor(image.size)
        coef = torch.hstack((shape, shape)) / 640
        img = image.convert("RGB").resize((640, 640))
        img_tensor = self.transform(img).unsqueeze(0).to(self.device)
        pred, boxes, scores = self.model(img_tensor, conf_thres = conf_threshold)
        selected = torchvision.ops.nms(boxes, scores, nms_threshold)
        predictions_new = list()
        for i in selected:
            #remove prob and class
            pred_i = torch.Tensor(pred[i][:4])
            #Loop through coordinates
            for j in range(4):
                #If any are negative, map to 0
                if pred_i[j] < 0:
                    pred_i[j] = 0
            #multiply by coef
            pred_i *= coef
            predictions_new.append(pred_i)
        predictions_new = torch.stack(predictions_new, dim=0)
        return predictions_new