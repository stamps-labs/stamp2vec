import torch
from torchvision import transforms
from huggingface_hub import hf_hub_download

class ViTStamp():
    def __init__(self):
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = torch.jit.load(hf_hub_download(repo_id="stamps-labs/vits8-stamp", filename="vits8stamp-torchscript.pth"))
        self.transform = transforms.ToTensor()
    def __call__(self, image) -> torch.Tensor():
        img_tensor = self.transform(image).cuda().unsqueeze(0) if self.device == "cuda" else self.transform(image).unsqueeze(0)
        features = self.model(img_tensor)
        return features