from embedding_models.vae.model import Encoder
from huggingface_hub import hf_hub_download
import torch
import numpy as np
from torchvision.transforms.functional import to_tensor

class VaePipeline:
    def __init__(self):
        self.encoder = Encoder()
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

    @classmethod
    def from_pretrained(cls, model_path_hf: str = None, filename_hf: str = None, local_model_path: str = None):
        vae = cls()
        if model_path_hf is not None and filename_hf is not None:
            vae.encoder.load_state_dict(torch.load(hf_hub_download(model_path_hf, filename_hf), map_location='cpu'))
            vae.encoder.to(vae.device)
            vae.encoder.eval()
        elif local_model_path is not None:
            vae.encoder.load_state_dict(torch.load(local_model_path, map_location='cpu'))
            vae.encoder.to(vae.device)
            vae.encoder.eval()
        return vae
    
    def __call__(self, image) -> torch.Tensor:
        image = image.convert("RGB")
        img_tensor = to_tensor(image.resize((118, 118)))
        return np.array(self.encoder(img_tensor.unsqueeze(0).to(self.device))[0][0].detach().cpu())