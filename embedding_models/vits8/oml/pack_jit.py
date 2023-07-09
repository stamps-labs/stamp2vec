import torch
from model_oml import EmbeddingModelOML
from huggingface_hub import HfApi
import argparse

parser = argparse.ArgumentParser("Packing checkpoint to JIT and serving to HF repo",
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--upload-to-hf", action="store_true", help="Whether to upload model to hf hub, REQUIRES LOGGING")
parser.add_argument("--path-to-save", help="Where to save the model file", default="../models/")
parser.add_argument("--model-name", help="Which model name to save in folder", default="vits8stamp-torchscript.pth")
parser.add_argument("--repo-id", help="repository id on huggingface", default="stamps-labs/vits8-stamp")

args = parser.parse_args()
config = vars(args)

if __name__ == "__main__":
    model = EmbeddingModelOML().extractor.cuda()

    model.eval()

    with torch.no_grad():
        model_ts = torch.jit.script(model)

    model_ts.save(config["path_to_save"]+config["model_name"])
    if config["upload_to_hf"]: 
        api = HfApi()
        api.upload_file(
            path_or_fileobj=config["path_to_save"]+config["model_name"],
            path_in_repo=config["model_name"],
            repo_id=config["repo_id"],
            repo_type="model"
            )
    