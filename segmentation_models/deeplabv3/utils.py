from pathlib import Path

from torch.utils.data import DataLoader
from torchvision import transforms

from data import SegmentationDataset


def get_dataloader_single_folder(data_dir: str,
                                 image_folder: str = 'images',
                                 mask_folder: str = 'masks',
                                 fraction: float = 0.2,
                                 batch_size: int = 4):
    """Create train and test dataloader from a single directory containing
    the image and mask folders.

    Args:
        data_dir (str): Data directory path or root
        image_folder (str, optional): Image folder name. Defaults to 'Images'.
        mask_folder (str, optional): Mask folder name. Defaults to 'Masks'.
        fraction (float, optional): Fraction of Test set. Defaults to 0.2.
        batch_size (int, optional): Dataloader batch size. Defaults to 4.

    Returns:
        dataloaders: Returns dataloaders dictionary containing the
        Train and Test dataloaders.
    """
    data_transforms = transforms.Compose([transforms.Resize((224, 224), interpolation=transforms.InterpolationMode.NEAREST), transforms.ToTensor()])

    image_datasets = {
        x: SegmentationDataset(data_dir,
                               image_folder=image_folder,
                               mask_folder=mask_folder,
                               seed=100,
                               fraction=fraction,
                               subset=x,
                               transforms=data_transforms)
        for x in ['Train', 'Test']
    }
    dataloaders = {
        x: DataLoader(image_datasets[x],
                      batch_size=batch_size,
                      shuffle=True,
                      num_workers=0)
        for x in ['Train', 'Test']
    }
    return dataloaders


def iou(gt_mask, pred_mask, threshold):
    
    pred_mask = (pred_mask > threshold) * 1
    gt_mask = (gt_mask == 1) * 1

    overlap = pred_mask * gt_mask  # Logical AND
    union = (pred_mask + gt_mask)>0  # Logical OR
    iou = overlap.sum() / float(union.sum())
    return iou