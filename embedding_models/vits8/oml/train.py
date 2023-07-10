import pytorch_lightning as pl
import torch
import pandas as pd

from oml.datasets.base import DatasetQueryGallery, DatasetWithLabels
from oml.lightning.modules.extractor import ExtractorModule
from oml.lightning.callbacks.metric import MetricValCallback
from oml.losses.triplet import TripletLossWithMiner
from oml.metrics.embeddings import EmbeddingMetrics
from oml.miners.inbatch_all_tri import AllTripletsMiner
from oml.models.vit.vit import ViTExtractor
from oml.samplers.balance import BalanceSampler
from pytorch_lightning.loggers import TensorBoardLogger

import argparse

parser = argparse.ArgumentParser("Train model with OML", 
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument("--root-dir", help="root directory for train data", default="data/train_val/")
parser.add_argument("--train-dataframe-name", help="name of dataframe in root directory", default="df_stamps.csv")
parser.add_argument("--train-images", help="name of directory with images", default="images/")
parser.add_argument("--num-epochs", help="number of epochs to train model", default=100)
parser.add_argument("--model-arch", help="which model architecture to use, check model zoo", default="vits8")
parser.add_argument("--weights", 
                    help="""
                    pretrained weights for model, choose from model zoo 
                    https://open-metric-learning.readthedocs.io/en/latest/feature_extraction/zoo.html
                    """, 
                    default="vits8_dino")
parser.add_argument("--checkpoint", help="resume training from checkpoint, provide path", default=None)
parser.add_argument("--num-labels", help="number of labels in dataset, set less if cuda out of memory", default=6)
parser.add_argument("--num-instances", help="number of instances for each label in batch, set less if cuda out of memory", default=2)
parser.add_argument("--val-batch-size", help="batch size for validation", default=4)
parser.add_argument("--log-data", action="store_true", help="Whether to log data")

args = parser.parse_args()
config = vars(args)

dataset_root = config['root_dir']
df = pd.read_csv(f"{dataset_root}{config['train_dataframe_name']}")

df_train = df[df["split"] == "train"].reset_index(drop=True)
df_val = df[df["split"] == "validation"].reset_index(drop=True)
df_val["is_query"] = df_val["is_query"].astype(bool)
df_val["is_gallery"] = df_val["is_gallery"].astype(bool)

extractor = ViTExtractor(config['weights'], arch=config['model_arch'], normalise_features=False)

optimizer = torch.optim.SGD(extractor.parameters(), lr=1e-6)
train_dataset = DatasetWithLabels(df_train, dataset_root=dataset_root)
criterion = TripletLossWithMiner(margin=0.1, miner=AllTripletsMiner())
batch_sampler = BalanceSampler(train_dataset.get_labels(), n_labels=config['num_labels'], n_instances=config['num_instances'])
train_loader = torch.utils.data.DataLoader(train_dataset, batch_sampler=batch_sampler)

val_dataset = DatasetQueryGallery(df_val, dataset_root=dataset_root)
val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=config['val_batch_size'])
metric_callback = MetricValCallback(metric=EmbeddingMetrics(extra_keys=[train_dataset.paths_key,], cmc_top_k=(5, 3, 1)), log_images=True)

if config['log_data']:
    logger = TensorBoardLogger(".")

pl_model = ExtractorModule(extractor, criterion, optimizer)
trainer = pl.Trainer(max_epochs=config['num_epochs'], callbacks=[metric_callback], num_sanity_val_steps=0, accelerator='gpu', devices=1, resume_from_checkpoint=config['checkpoint'])
trainer.fit(pl_model, train_dataloaders=train_loader, val_dataloaders=val_loader)