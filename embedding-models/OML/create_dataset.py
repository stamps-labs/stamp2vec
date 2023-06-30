import os
from PIL import Image
import pandas as pd

import argparse

parser = argparse.ArgumentParser("Create a dataset for training with OML", 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)

parser.add_argument("--root-data-path", help="Path to images for dataset", default="data/train_val/")
parser.add_argument("--image-data-path", help="Image folder in root data path", default="images/")
parser.add_argument("--train-val-split", 
                    help="In which ratio to split data in format train:val (For example 80:20)", default="80:20")
parser.add_argument("--separator", 
                    help="What separator is used in image name to separate class name and instance (E.g. circle1_5, separator=_)",
                    default="_")

args = parser.parse_args()
config = vars(args)

root_path = config["root_data_path"]
image_path = config["image_data_path"]
separator = config["separator"]

train_prc, val_prc = tuple(int(num)/100 for num in config["train_val_split"].split(":"))

class_names = set()
for image in os.listdir(root_path+image_path):
    if image.endswith(("png", "jpg", "bmp", "webp")):
        img_name = image.split(".")[0]
        Image.open(root_path+image_path+image).resize((224,224)).save(root_path+image_path+img_name+".png", "PNG")
        if not image.endswith("png"):
            os.remove(root_path+image_path+image)
        img_name = img_name.split(separator)
        class_name = img_name[0]+img_name[1]
        class_names.add(class_name)
    else:
        print("Not all of the images are in supported format")
    

#For each class in set assign its index in a set as a class label.
class_label_dict = {}
for ind, name in enumerate(class_names):
    class_label_dict[name] = ind

class_count = len(class_names)
train_class_count = int(class_count*train_prc)
print(train_class_count)

df_dict = {"label": [],
           "path": [],
           "split": [],
           "is_query": [],
           "is_gallery": []}
for image in os.listdir(root_path+image_path):
    if image.endswith((".png", ".jpg", ".bmp", ".webp")):
        img_name = image.split(".")[0].split(separator)
        class_name = img_name[0]+img_name[1]
        label = class_label_dict[class_name]
        path = image_path+image
        split = "train" if label <= train_class_count else "validation"
        is_query, is_gallery = (1, 1) if split=="validation" else (None, None)
        df_dict["label"].append(label)
        df_dict["path"].append(path)
        df_dict["split"].append(split)
        df_dict["is_query"].append(is_query)
        df_dict["is_gallery"].append(is_gallery)

df = pd.DataFrame(df_dict)

df.to_csv(root_path+"df_stamps.csv", index=False)