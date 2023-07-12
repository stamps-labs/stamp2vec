import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
from sklearn.model_selection import train_test_split
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from PIL import Image

from pathlib import Path
from random import randint

from utils import *

"""
    Dataset class for storing stamps data.
    
    Arguments:
    data -- list of dictionaries containing file_path (path to the image), box_nb (number of boxes on the image), and boxes of shape (4,)
    image_folder -- path to folder containing images
    transforms -- transforms from albumentations package
"""
class StampDataset(Dataset):
    def __init__(
            self, 
            data=read_data(), 
            image_folder=Path(IMAGE_FOLDER),
            transforms=None):
        self.data = data
        self.image_folder = image_folder
        self.transforms = transforms

    def __getitem__(self, idx):
        item = self.data[idx]
        image_fn = self.image_folder / item['file_path']
        boxes = item['boxes']
        box_nb = item['box_nb']
        labels = torch.zeros((box_nb, 2), dtype=torch.int64)
        labels[:, 0] = 1

        img = np.array(Image.open(image_fn))

        try:
            if self.transforms:
                sample = self.transforms(**{
                    "image":img,
                    "bboxes": boxes,
                    "labels": labels,
                })
                img = sample['image']
                boxes = torch.stack(tuple(map(torch.tensor, zip(*sample['bboxes'])))).permute(1, 0)
        except:
            return self.__getitem__(randint(0, len(self.data)-1))

        target_tensor = boxes_to_tensor(boxes.type(torch.float32))
        return img, target_tensor

    def __len__(self):
        return len(self.data)

def collate_fn(batch):
    return tuple(zip(*batch))


def get_datasets(data_path=ANNOTATIONS_PATH, train_transforms=None, val_transforms=None):
    """
        Creates StampDataset objects.

        Arguments: 
        data_path -- string or Path, specifying path to annotations file
        train_transforms -- transforms to be applied during training
        val_transforms -- transforms to be applied during validation

        Returns:
        (train_dataset, val_dataset) -- tuple of StampDataset for training and validation
    """
    data = read_data(data_path)
    if train_transforms is None:
        train_transforms = A.Compose([
            A.RandomCropNearBBox(max_part_shift=0.6, p=0.4),
            A.Resize(height=448, width=448),
            A.HorizontalFlip(p=0.5),
            A.VerticalFlip(p=0.5),
            # A.Affine(scale=(0.9, 1.1), translate_percent=(0.05, 0.1), rotate=(-45, 45), shear=(-30, 30), p=0.3),
            # A.Blur(blur_limit=4, p=0.3),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ],
        bbox_params={
            "format":"coco",
            'label_fields': ['labels']
        })

    if val_transforms is None:
        val_transforms = A.Compose([
            A.Resize(height=448, width=448),
            A.Normalize(),
            ToTensorV2(p=1.0),
        ],
        bbox_params={
            "format":"coco",
            'label_fields': ['labels']
        })
    train, test_data = train_test_split(data, test_size=0.1, shuffle=True)

    train_data, val_data = train_test_split(train, test_size=0.2, shuffle=True)

    train_dataset = StampDataset(train_data, transforms=train_transforms)
    val_dataset = StampDataset(val_data, transforms=val_transforms)
    test_dataset = StampDataset(test_data, transforms=val_transforms)

    return train_dataset, val_dataset, test_dataset


def get_loaders(batch_size=8, data_path=ANNOTATIONS_PATH, num_workers=0, train_transforms=None, val_transforms=None):
    """
        Creates StampDataset objects.

        Arguments: 
        batch_size -- integer specifying the number of images in the batch
        data_path -- string or Path, specifying path to annotations file
        train_transforms -- transforms to be applied during training
        val_transforms -- transforms to be applied during validation

        Returns:
        (train_loader, val_loader) -- tuple of DataLoader for training and validation
    """
    train_dataset, val_dataset, _ = get_datasets(data_path)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=collate_fn, drop_last=True)

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn)
    
    return train_loader, val_loader
