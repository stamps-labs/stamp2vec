import torch
from torchvision import transforms
from huggingface_hub import hf_hub_download
import numpy as np

class Vits8Pipeline:
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = None # Initialized upon loading torchscript
        self.transform = transforms.ToTensor()
    
    @classmethod
    def from_pretrained(cls, model_path_hf: str = None, filename_hf: str = None, local_model_path: str = None):
        vit = cls()
        if model_path_hf is not None and filename_hf is not None:
            vit.model = torch.jit.load(hf_hub_download(model_path_hf, filename=filename_hf), map_location='cpu')  
            vit.model.to(vit.device)
            vit.model.eval()      
        elif local_model_path is not None:
            vit.model = torch.jit.load(local_model_path, map_location='cpu')
            vit.model.to(vit.device)
            vit.model.eval()
        return vit

    def __call__(self, image) -> torch.Tensor:
        image = image.convert("RGB")
        img_tensor = self.transform(image).to(self.device)
        features = np.array(self.model(img_tensor)[0].detach().cpu())
        return features