from oml.models.vit.vit import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
import torch
from PIL import Image

class EmbeddingModelOML:
    def __init__(self, model_path: str = "../models/vits8-stamp.ckpt", arch: str = "vits8", normalise_features: bool = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.extractor = ViTExtractor(model_path, arch=arch, normalise_features=normalise_features).to(self.device)
        self.transform, _ = get_transforms_for_pretrained("vits8_dino")
    def __call__(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self.transform(image).cuda().unsqueeze(0) if self.device == "cuda" else self.transform(image).unsqueeze(0)
        features = self.extractor(img_tensor)
        return features
