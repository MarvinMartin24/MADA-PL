import os
import pytorch_lightning as pl
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.datasets as dset

from tools.utils import get_train_val_test_src, get_train_test_tgts, download_and_extract_office31

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

    def __init__(self, cfg, experiment_path):
        super().__init__()
        self.cfg = cfg
        self.experiment_path = experiment_path

    def setup(self):
        
        if self.cfg['input']['dataset']['src'] in ['AMAZON', 'DSLR', 'WEBCAM']:
            for tgt in self.cfg['input']['dataset']['tgts']:
                assert(tgt in ['AMAZON', 'DSLR', 'WEBCAM']), 'Wrong dataset name provided in config file. Please only use AMAZON, DSLR, WEBCAM.'
            print("Downloading Office31 Dataset...")
            download_and_extract_office31(self.cfg, self.experiment_path)
        
        elif self.cfg['input']['dataset']['src'] in ['MNIST', 'MNISTM']:
            print("Download MNIST Dataset..") 
            for tgt in self.cfg['input']['dataset']['tgts']:
                assert(tgt in ['MNIST', 'MNISTM']), 'Wrong dataset name provided in config file. Please only use MNIST, MNISTM.'
        else:
            raise Exception("Wrong dataset name provided in config file. Please only use AMAZON, DSLR, WEBCAM, MNIST, MNISTM.")



        # Load Train/Val of the Source (train_set_src and val_set_src)
        self.train_set_src, self.val_set_src, self.test_set_src = get_train_val_test_src(self.cfg, self.experiment_path)
        
        self.classes = self.train_set_src.dataset.classes
        self.domains = [self.cfg['input']['dataset']['src']] + self.cfg['input']['dataset']['tgts']
        self.num_classes =len(self.train_set_src.dataset.classes)
        self.num_domains = len(self.domains)
        
        print(f"CLASSES:{self.classes}, DOMAINS:{self.domains}")


        # Load Train of the Targets (not val because we only evaluate the class classifier but not the domain classifiers)
        self.train_set_tgts, self.test_set_tgts = get_train_test_tgts(self.cfg, self.experiment_path) # is a list of torch Datasets

        # Get the size of the dataloader for the loss later
        min_len_dataloader_tgts = min([len(tgt.dataset) for tgt in self.train_set_tgts])
        self.len_dataloader = min(len(self.train_set_src), min_len_dataloader_tgts)


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
        concat_test_set = ConcatDataset(
            self.test_set_src,
            *self.test_set_tgts
        )
        
        return DataLoader(
            concat_test_set,
            shuffle=False,
            batch_size=self.cfg["training"]["batch_size"],
            num_workers=self.cfg["training"]["num_workers"])
