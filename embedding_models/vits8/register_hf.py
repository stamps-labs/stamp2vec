from transformers import PretrainedConfig, PreTrainedModel, AutoModel, AutoConfig
from transformers.configuration_utils import PretrainedConfig
from model_torch import EmbeddingModelTorch
import torch
from torchvision import transforms



class ViTStampConfig(PretrainedConfig):
    model_type = "vits8stamp"
    def __init__(self, **kwargs):
        super().__init__(**kwargs)

class ViTStamp(PreTrainedModel):
    config_class = ViTStampConfig
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.transform = transforms.ToTensor()
        self.model = torch.jit.load("models/vits8stamp-torchscript.pth")
    def forward(self, image):
        img_tensor = self.transform(image).cuda().unsqueeze(0) if self.device == "cuda" else self.transform(image).unsqueeze(0)
        features = self.model(img_tensor)
        return features
    
if __name__ == "__main__":
    AutoConfig.register("vits8stamp", ViTStampConfig)
    AutoModel.register(ViTStampConfig, ViTStamp)
    vit_config = ViTStampConfig()
    vit_model = ViTStamp(vit_config)
    vit_model.push_to_hub("stamps-labs/vits8-stamp")
