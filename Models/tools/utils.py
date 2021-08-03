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
import datasetops as do
class ImagePredictionLogger(pl.Callback):
    def __init__(self, dataModule, val_samples, title, num_samples=32):
        super().__init__()
        self.dataModule = dataModule
        self.val_imgs, self.val_labels = val_samples
        self.val_imgs = self.val_imgs[:num_samples]
        self.val_labels = self.val_labels[:num_samples]
        self.title = title

    def on_validation_epoch_end(self, trainer, pl_module):

        classes = self.dataModule.classes
        domains = ["MNIST", "MNISTM"]
        val_imgs = self.val_imgs.to(device=pl_module.device)
        class_pred_logit, domain_pred = pl_module(val_imgs)
        class_pred_prob = torch.argmax(F.softmax(class_pred_logit, dim=1), dim=1)
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
"""
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"

    session = requests.Session()

    response = session.get(URL, params={"id": id}, stream=True)
    token = get_confirm_token(response)

    if token:
        params = {"id": id, "confirm": token}
        response = session.get(URL, params=params, stream=True)

    save_response_content(response, destination)


def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith("download_warning"):
            return value

    return None


def save_response_content(response, destination):
    CHUNK_SIZE = 200000

    print("Downloading")
    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk:  # filter out keep-alive new chunks
                f.write(chunk)


def download_and_extract_office31(cfg):

    if os.path.isdir(os.path.join(cfg['output']['save_path'], 'office31')):
        return "Done"


    download_file_from_google_drive(
        id="0B4IapRTv9pJ1WGZVd1VDMmhwdlE", destination=str(os.path.join(cfg['output']['save_path'], 'office31'))
    )

    target_path.mkdir(parents=True, exist_ok=True)

    print("Unpacking")
    shutil.unpack_archive(str(cfg['output']['save_path']), target_path)

    print("Cleaning up")
    #tmp_path.unlink()

    print("Done")

 """
def download_and_extract_office31(cfg):
     office31(
        source_name = "dslr",
        target_name = "amazon",
        seed=cfg['seed'],
        same_to_diff_class_ratio=3,
        image_resize=(cfg['input']['dataset']['transformation']['img_size'], cfg['input']['dataset']['transformation']['img_size']),
        group_in_out=True, # groups data: ((img_s, img_t), (lbl_s, _lbl_t))
        framework_conversion="pytorch",
        office_path = str(os.path.join(cfg['output']['save_path'], "office31")) #automatically downloads to "~/data"
    )   

def get_train_val_test_src(cfg):
    print(cfg['input']['dataset']['src'])
    if cfg['input']['dataset']['src'] == "MNIST":
        transform_mnist_train = get_transform(cfg["input"]["dataset"]['transformation'])
        dataset = torchvision.datasets.MNIST(root=cfg['output']['save_path'], train=True, download=True, transform=transform_mnist_train)
        train_set, val_set = torch.utils.data.random_split(dataset, [50000, 10000])
        test_set = torchvision.datasets.MNIST(root=cfg['output']['save_path'], train=False, download=True, transform=transform_mnist_train)
    
    elif cfg['input']['dataset']['src'] == "AMAZON":
        transform_amazon_train = get_transform(cfg["input"]["dataset"]["transformation"])
        dataset = ImageFolder(root=os.path.join(cfg['output']['save_path'],'office31/amazon/images'), transform=transform_amazon_train)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [2253, 282, 282]) 

    
    elif cfg['input']['dataset']['src'] == "WEBCAM":
        transform_webcam_train = get_transform(cfg["input"]["dataset"]["transformation"])
        dataset = ImageFolder(root=os.path.join(cfg['output']['save_path'],'office31/webcam/images'), transform=transform_webcam_train)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [635, 80, 80])

    elif cfg['input']['dataset']['src'] == "DSLR":
        transform_dslr_train = get_transform(cfg["input"]["dataset"]["transformation"])
        dataset = ImageFolder(root=os.path.join(cfg['output']['save_path'],'office31/dslr/images'), transform=transform_dslr_train)
        train_set, val_set, test_set = torch.utils.data.random_split(dataset, [398, 50, 50])


    else:
        print('Source dataset name does not exist')
    return train_set, val_set, test_set

def get_train_test_tgts(cfg):
    train_sets = []
    test_sets = []
    for tgt in cfg['input']['dataset']['tgts']:
        print(tgt)
        if tgt == "MNISTM":
            transform_mnistm_train = get_transform(cfg["input"]["dataset"]['transformation'])
            dataset = MNISTM(root=cfg['output']['save_path'], train=True, download=True, transform=transform_mnistm_train)
            train_set_tgt, _ = torch.utils.data.random_split(dataset, [50000, 10000])
            test_set_tgt = MNISTM(root=cfg['output']['save_path'], train=False, download=True, transform=transform_mnistm_train)
            train_sets.append(train_set_tgt)
            test_sets.append(test_set_tgt)
        elif tgt == "AMAZON":
            transform_amazon_train = get_transform(cfg["input"]["dataset"]['transformation'])
            dataset = ImageFolder(root=os.path.join(cfg['output']['save_path'],'office31/amazon/images'), transform=transform_amazon_train)
            train_set_tgt, test_set_tgt = torch.utils.data.random_split(dataset, [2535, 282])
            train_sets.append(train_set_tgt)
            test_sets.append(test_set_tgt)
        elif tgt == "DSLR":
            transform_dslr_train = get_transform(cfg["input"]["dataset"]['transformation'])
            dataset = ImageFolder(root=os.path.join(cfg['output']['save_path'],'office31/dslr/images'), transform=transform_dslr_train)
            train_set_tgt, test_set_tgt = torch.utils.data.random_split(dataset, [448, 50])
            train_sets.append(train_set_tgt)
            test_sets.append(test_set_tgt)
        elif tgt == "WEBCAM":
            transform_webcam_train = get_transform(cfg["input"]["dataset"]['transformation'])
            dataset = ImageFolder(root=(cfg['output']['save_path'],'office31/webcam/images'), transform=transform_webcam_train)
            train_set_tgt, test_set_tgt = torch.utils.data.random_split(dataset, [715, 80])
            train_sets.append(train_set_tgt)
            test_sets.append(test_set_tgt)



        else:
            raise Exception("Target dataset name does not exist")


    return train_sets, test_sets

