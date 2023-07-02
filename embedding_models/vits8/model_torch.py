from PIL import Image
import torch
from torchvision import transforms

class EmbeddingModelTorch:
    def __init__(self, model_path: str = "models/vits8stamp-torchscript.pth"):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load(model_path).to(self.device)
        self.transform = transforms.ToTensor()
    def __call__(self, image: Image.Image) -> torch.Tensor:
        img_tensor = self.transform(image).cuda().unsqueeze(0) if self.device == "cuda" else self.transform(image).unsqueeze(0)
        features = self.model(img_tensor)
        return features