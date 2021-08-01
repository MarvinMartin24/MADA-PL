import torch
import numpy as np
import torchvision
import pytorch_lightning as pl
import wandb
from torchvision import transforms

from tools.MNISTM import MNISTM

class ImagePredictionLogger(pl.Callback):
    def __init__(self, val_samples, title, num_samples=32):
        super().__init__()
        self.val_imgs, self.val_labels = next(val_samples)
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]
        self.title = title

    def on_validation_epoch_end(self, trainer, pl_module):

        classes = [""]
        domains = [""]
        val_imgs = self.val_imgs.to(device=pl_module.device)
        class_pred_logit, domain_pred = pl_module(val_imgs)
        class_pred_prob = torch.sigmoid(class_pred_logit)
        domain_pred = torch.argmax(domain_pred, dim=1)
        trainer.logger.experiment.log({
            self.title : [
                wandb.Image(x, caption=f"Preds: {round(pred.item(), 3)} | {domains[domain]}\nLabel: {classes[y]}")
                    for x, pred, domain, y in zip(val_imgs, class_pred_prob, domain_pred, self.val_labels)
            ],
        })

def get_transform(transformation):
    return transforms.Compose([
        transforms.Resize(transformation['img_size']),
        transforms.Grayscale(num_output_channels=3),
        transforms.ToTensor(),
        transforms.Normalize(transformation['mean'], transformation['std']),
    ])

def get_train_val_src(cfg):
    if cfg['input']['dataset']['src'] == "MNIST":
        transform_mnist_train = get_transform(cfg["input"]["dataset"]['transformation'])
        dataset = torchvision.datasets.MNIST(root=cfg['output']['save_path'], train=True, download=True, transform=transform_mnist_train)
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
    else:
        #train_set, val_set = #TODO
        pass
    return train_set, val_set

def get_train_tgts(cfg):
    train_sets = []
    for tgt in cfg['input']['dataset']['tgts']:
        if tgt == "MNISTM":
            transform_mnistm_train = get_transform(cfg["input"]["dataset"]['transformation'])
            dataset = MNISTM(root=cfg['output']['save_path'], train=True, download=True, transform=transform_mnistm_train)
            train_set_tgt, _ = torch.utils.data.random_split(dataset, [50000, 10000])
            train_sets.append(train_set_tgt)

        else:
            #TODO
            # train_sets.append(train_set_tgt)
            pass

    return train_sets

def get_test_tgts(cfg):
    test_sets = []
    for tgt in cfg['input']['dataset']['tgts']:

        if tgt == "MNIST":
            transform_mnist_train = get_transform(cfg["input"]["dataset"]['transformation'])
            test_set_tgt = torchvision.datasets.MNIST(root=cfg['output']['save_path'], train=False, download=True, transform=transform_mnist_train)
            test_sets.append(test_set_tgt)

        if tgt == "MNISTM":
            transform_mnistm_train = get_transform(cfg["input"]["dataset"]['transformation'])
            test_set_tgt = MNISTM(root=cfg['output']['save_path'], train=False, download=True, transform=transform_mnistm_train)
            test_sets.append(test_set_tgt)

        else:
            #TODO
            #train_sets.append(train_set_tgt)
            pass
    return test_sets
