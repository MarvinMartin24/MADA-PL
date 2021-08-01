import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import wandb
import os

from nets.GRL import GRL
from nets.backbones import load_backbone


class MADA(pl.LightningModule):

    def __init__(self, cfg, mode, dataModule=None):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
