from typing import Any
import torch
from huggingface_hub import hf_hub_download
from PIL import Image
import torchvision.transforms as transforms
import numpy as np

class DeepLabv3Pipeline:

    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.transforms = transforms.Compose(
            [
                transforms.Resize((336, 336), interpolation=transforms.InterpolationMode.NEAREST),
                transforms.ToTensor()
            ]
        )
        self.model = None

    @classmethod
    def from_pretrained(cls, model_path_hf: str = None, filename_hf: str = "weights.pt", local_model_path: str = None):
        dl = cls()
        if model_path_hf is not None and filename_hf is not None:
            dl.model = torch.load(hf_hub_download(model_path_hf, filename=filename_hf), map_location='cpu')
            dl.model.to(dl.device)
            dl.model.eval()
        elif local_model_path is not None:
            dl.model = torch.load(local_model_path, map_location='cpu')
            dl.model.to(dl.device)
            dl.model.eval()
        return dl

    def __call__(self, image: Image.Image, threshold: float = 0.4) -> Image.Image:
        image = image.convert("RGB")
        output = self.model(self.transforms(image).unsqueeze(0).to(self.device))
        return  Image.fromarray((255 * np.where(output['out'][0].permute(1, 2, 0).detach().cpu() > threshold, 
                                                self.transforms(image).permute(1, 2, 0), 1)).astype(np.uint8))
