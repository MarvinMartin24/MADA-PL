import torch
import numpy as np
import torchvision
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
from torchvision import transforms
from tools.MNISTM import MNISTM
import requests
from pathlib import Path
import shutil
import os
from torchvision.datasets import ImageFolder
from office31 import office31

def get_transform(transformation, domain):
    if transformation[domain] == 'transform_GS_DA':
        return transform_GS(transformation)
    elif transformation[domain] == 'transform_RGB_DA':
        return transform_RGB(transformation)
    elif transformation[domain] == 'transform_GS':
        return transform_GS(transformation)
    elif transformation[domain] == 'transform_RGB':
        return transform_RGB(transformation)
    elif transformation[domain] == 'transform_mnist':
        return transform_mnist(transformation)
    elif transformation[domain] == 'transform_mnistm':
        return transform_mnistm(transformation)
    raise Exception('Name of transforms given in config not correct')

def transform_mnist(transformation):
    return transforms.Compose([
        transforms.CenterCrop(size=(transformation['img_size'], transformation['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
        transforms.Lambda(lambda x: x.repeat(3, 1, 1))
    ])

def transform_mnistm(transformation):
    return transforms.Compose([
        transforms.CenterCrop(size=(transformation['img_size'], transformation['img_size'])),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5]),
    ])

def transform_GS_DA(transformation):
    return transforms.Compose([
        transforms.Resize(transformation['img_size']),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize(transformation['mean'], transformation['std']),
    ])

def transform_GS(transformation):
    return transforms.Compose([
        transforms.Resize(transformation['img_size']),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(transformation['mean'], transformation['std']),
    ])

def transform_RGB_DA(transformation):
    return transforms.Compose([
        transforms.Resize(transformation['img_size']),
        transforms.Grayscale(num_output_channels=3),
        transforms.RandomHorizontalFlip(0.3),
        transforms.RandomRotation(0.3),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3),
        transforms.ToTensor(),
        transforms.Normalize(transformation['mean'], transformation['std']),
    ])

def transform_RGB(transformation):
    return transforms.Compose([
        transforms.Resize(transformation['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(transformation['mean'], transformation['std']),
    ])

def download_and_extract_office31(cfg):
     office31(
        source_name = "dslr",
        target_name = "amazon",
        seed=cfg['seed'],
        same_to_diff_class_ratio=3,
        image_resize=(cfg['input']['dataset']['transformation']['img_size'], cfg['input']['dataset']['transformation']['img_size']),
        group_in_out=True,
        framework_conversion="pytorch",
        office_path = str(os.path.join(cfg['output']['save_path'], "office31")) #automatically downloads to "~/data"
    )   

def get_train_val_test_src(cfg):
   
    if cfg['input']['dataset']['src'] == "MNIST":
        transform_mnist_train = get_transform(cfg["input"]["dataset"]['transformation'], 'src')
        dataset = torchvision.datasets.MNIST(root=cfg['output']['save_path'], train=True, download=True, transform=transform_mnist_train)
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
        test_set = torchvision.datasets.MNIST(root=cfg['output']['save_path'], train=False, download=True, transform=transform_mnist_train)
    
    elif cfg['input']['dataset']['src'] == "AMAZON":
        transform_amazon_train = get_transform(cfg["input"]["dataset"]["transformation"], 'src')
        dataset = ImageFolder(root=os.path.join(cfg['output']['save_path'],'office31/amazon/images'), transform=transform_amazon_train)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [2253, 282, 282]) 

    
    elif cfg['input']['dataset']['src'] == "WEBCAM":
        transform_webcam_train = get_transform(cfg["input"]["dataset"]["transformation"], 'src')
        dataset = ImageFolder(root=os.path.join(cfg['output']['save_path'],'office31/webcam/images'), transform=transform_webcam_train)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [635, 80, 80])

    elif cfg['input']['dataset']['src'] == "DSLR":
        transform_dslr_train = get_transform(cfg["input"]["dataset"]["transformation"], 'src')
        dataset = ImageFolder(root=os.path.join(cfg['output']['save_path'],'office31/dslr/images'), transform=transform_dslr_train)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [398, 50, 50])


    else:
        raise Exception('Source dataset name does not exist')
    return train_set, val_set, test_set

def get_train_test_tgts(cfg):
    train_sets = []
    test_sets = []
    for tgt in cfg['input']['dataset']['tgts']:
        
        if tgt == "MNISTM":
            transform_mnistm_train = get_transform(cfg["input"]["dataset"]['transformation'], 'tgt')
            dataset = MNISTM(root=cfg['output']['save_path'], train=True, download=True, transform=transform_mnistm_train)
            train_set_tgt, _ = torch.utils.data.random_split(dataset, [50000, 10000])
            test_set_tgt = MNISTM(root=cfg['output']['save_path'], train=False, download=True, transform=transform_mnistm_train)
            train_sets.append(train_set_tgt)
            test_sets.append(test_set_tgt)
        elif tgt == "AMAZON":
            transform_amazon_train = get_transform(cfg["input"]["dataset"]['transformation'], 'tgt')
            dataset = ImageFolder(root=os.path.join(cfg['output']['save_path'],'office31/amazon/images'), transform=transform_amazon_train)
            train_set_tgt, test_set_tgt = torch.utils.data.random_split(dataset, [2535, 282])
            train_sets.append(train_set_tgt)
            test_sets.append(test_set_tgt)
        elif tgt == "DSLR":
            transform_dslr_train = get_transform(cfg["input"]["dataset"]['transformation'], 'tgt')
            dataset = ImageFolder(root=os.path.join(cfg['output']['save_path'],'office31/dslr/images'), transform=transform_dslr_train)
            train_set_tgt, test_set_tgt = torch.utils.data.random_split(dataset, [448, 50])
            train_sets.append(train_set_tgt)
            test_sets.append(test_set_tgt)
        elif tgt == "WEBCAM":
            transform_webcam_train = get_transform(cfg["input"]["dataset"]['transformation'], 'tgt')
            dataset = ImageFolder(root=(cfg['output']['save_path'],'office31/webcam/images'), transform=transform_webcam_train)
            train_set_tgt, test_set_tgt = torch.utils.data.random_split(dataset, [715, 80])
            train_sets.append(train_set_tgt)
            test_sets.append(test_set_tgt)

        else:
            raise Exception("Target dataset name does not exist")

    return train_sets, test_sets


class ImagePredictionLogger(pl.Callback):
    
    def __init__(self, dataModule, val_samples, title, num_samples=32):
        super().__init__()
        self.dataModule = dataModule
        self.title = title
        
        if self.title == "Test_Preds":
            (self.xs, self.ys), (self.xt, self.yt) = val_samples
            self.xt = self.xt[:num_samples]
            self.yt = self.yt[:num_samples]
            
        elif self.title == "Val_Preds":
            self.xs, self.ys = val_samples
            
        self.xs = self.xs[:num_samples]
        self.ys = self.ys[:num_samples]
        
    def on_validation_epoch_end(self, trainer, pl_module):

        classes = self.dataModule.classes
        domains = self.dataModule.domains
        
        self.xs = self.xs.to(device=pl_module.device)
        
        class_logit_src, domain_logit_src = pl_module(self.xs)
        class_pred_src = torch.argmax(F.softmax(class_logit_src, dim=1), dim=1)
        if isinstance(domain_logit_src, list):
            domain_pred_src = [torch.argmax(domain_logit_src[class_idx], dim=1) for class_idx in range(len(domain_logit_src))]
            trainer.logger.experiment.log({
                f"{self.title}_src" : [wandb.Image(x, caption=f"Pc(xs): {classes[p]} | Pd(xs):{domains[d[p]]}\nys: {classes[y]}")
                        for x, p, d, y in zip(self.xs, class_pred_src, domain_pred_src, self.ys)],
            })
        else:
            domain_pred_src = torch.argmax(domain_logit_src, dim=1)
            trainer.logger.experiment.log({
                f"{self.title}_src" : [wandb.Image(x, caption=f"Pc(xs): {classes[p]} | Pd(xs):{domains[d]}\nys: {classes[y]}")
                        for x, p, d, y in zip(self.xs, class_pred_src, domain_pred_src, self.ys)],
            })

        if self.title == "Test_Preds":
            self.xt = self.xt.to(device=pl_module.device)
            class_logit_tgt, domain_logit_tgt = pl_module(self.xt)
            class_pred_tgt = torch.argmax(F.softmax(class_logit_tgt, dim=1), dim=1)
            if isinstance(domain_logit_tgt, list):
                domain_pred_tgt = [torch.argmax(domain_logit_tgt[class_idx], dim=1) for class_idx in range(len(domain_logit_tgt))]
                trainer.logger.experiment.log({
                f"{self.title}_tqt" : [wandb.Image(x, caption=f"Pc(xt): {classes[p]} | Pd(xt):{domains[d[p]]}\nyt: {classes[y]}")
                        for x, p, d, y in zip(self.xt, class_pred_tgt, domain_pred_tgt, self.yt)],
                })
                
            else:
                domain_pred_tgt = torch.argmax(domain_logit_tgt, dim=1)
                trainer.logger.experiment.log({
                f"{self.title}_tqt" : [wandb.Image(x, caption=f"Pc(xt): {classes[p]} | Pd(xt):{domains[d]}\nyt: {classes[y]}")
                        for x, p, d, y in zip(self.xt, class_pred_tgt, domain_pred_tgt, self.yt)],
                })