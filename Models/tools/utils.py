import torch
import numpy as np
import torchvision
from torchvision import transforms

from tools.MNISTM import MNISTM

def get_transform(transformation):
    return transforms.Compose([
        transforms.Resize(transformation['img_size']),
        transforms.ToTensor(),
        transforms.Normalize(transformation['mean'], transformation['std'])
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
    return train_set, None

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
