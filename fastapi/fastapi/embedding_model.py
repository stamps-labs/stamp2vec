from oml.models.vit.vit import ViTExtractor
from oml.registry.transforms import get_transforms_for_pretrained
import torch
from PIL import Image

class EmbeddingModel:
    def __init__(self, model_path: str = "models/vits8-stamp.ckpt", arch: str = "vits8", normalise_features: bool = False):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = ViTExtractor(model_path, arch=arch, normalise_features=normalise_features).to(self.device)
        self.transform, self.im_reader = get_transforms_for_pretrained("vits8_dino") #Fix this later
    def predict(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self.transform(image).cuda().unsqueeze(0) if self.device == "cuda" else self.transform(image).unsqueeze(0)
        features = self.model(img_tensor)
        return features
