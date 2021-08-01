import os
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset

from tools.utils import get_train_val_src, get_train_tgts, get_test_tgts

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


class ConcatDataset(Dataset):
    def __init__(self, *datasets):
        self.datasets = datasets

    def __getitem__(self, i):
        return tuple(d[i] for d in self.datasets)

    def __len__(self):
        return min(len(d) for d in self.datasets)

class DataModule(pl.LightningDataModule):

    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg

    def setup(self):

        # Load Train/Val of the Source (train_set_src and val_set_src)
        self.train_set_src, self.val_set_src = get_train_val_src(self.cfg)


        # Load Train of the Targets (not val because we only evaluate the class classifier but not the domain classifiers)
        self.train_set_tgts = get_train_tgts(self.cfg) # is a list of torch Datasets

        # Get the size of the dataloader for the loss later
        min_len_dataloader_tgts = min([len(tgt.dataset) for tgt in self.train_set_tgts])
        self.len_dataloader = min(len(self.train_set_src), min_len_dataloader_tgts)

        # Load Test (should be from one of the Target distribution)
        self.test_set_tgts = get_test_tgts(self.cfg) # is a list of torch Datasets

    def train_dataloader(self):
        concat_dataset = ConcatDataset(
            self.train_set_src,
            *self.train_set_tgts
        )

        concat_loader = DataLoader(
            concat_dataset,
            batch_size=self.cfg["training"]["batch_size"],
            shuffle=True,
            num_workers=self.cfg["training"]["num_workers"],
            drop_last=True)

        self.len_dataloader = concat_dataset.__len__()
        return concat_loader

    def val_dataloader(self):
        return DataLoader(
            self.val_set_src,
            shuffle=False,
            batch_size=self.cfg["training"]["batch_size"],
            num_workers=self.cfg["training"]["num_workers"])

    def test_dataloader(self):
        concat_test_set = ConcatDataset(*self.test_set_tgts)
        return DataLoader(
            concat_test_set,
            shuffle=False,
            batch_size=self.cfg["training"]["batch_size"],
            num_workers=self.cfg["training"]["num_workers"])
