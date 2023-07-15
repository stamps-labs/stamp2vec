from pipelines.segmentation.deeplabv3 import DeepLabv3Pipeline
from PIL import Image

img = Image.open("dataset_generation/images/cropped_stamps/circle_2_5.png")

model = DeepLabv3Pipeline.from_pretrained("stamps-labs/deeplabv3-finetuned")

print(model(img))