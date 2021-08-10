import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict
import wandb
import os

from nets.GRL import GRL
from nets.loader import load_backbone, load_classifier


class DANN(pl.LightningModule):

    def __init__(self, cfg, mode, dataModule=None):
        super().__init__()
        self.cfg = cfg
        self.mode = mode
        
        if mode == 'Inference':
            self.num_classes = 31 if self.cfg['input']['dataset']['src'] in ['AMAZON', 'DSLR', 'WEBCAM'] else 10 #OFFICE31 OR MNIST
            self.backbone_fixed, _ = load_backbone(name=cfg['model']['backbone'], pretrained=cfg['model']['pretrained_backbone'])


        if mode == 'Train':

            self.len_dataloader = dataModule.len_dataloader
            self.classes = dataModule.classes
            self.num_classes = dataModule.num_classes
            self.criterion_class = nn.CrossEntropyLoss()
            self.criterion_domain = nn.CrossEntropyLoss()

            # log hyperparameters
            self.save_hyperparameters()

            # compute the accuracy
            self.train_accuracy_class =  pl.metrics.Accuracy().to(torch.device("cuda", 0))
            self.train_accuracy_domain_src =  pl.metrics.Accuracy().to(torch.device("cuda", 0))
            self.train_accuracy_domain_tgt =  pl.metrics.Accuracy().to(torch.device("cuda", 0))
            self.val_accuracy_class =  pl.metrics.Accuracy().to(torch.device("cuda", 0))
            self.test_accuracy_class_src =  pl.metrics.Accuracy().to(torch.device("cuda", 0))
            self.test_accuracy_class_tgt =  pl.metrics.Accuracy().to(torch.device("cuda", 0))


        self.backbone, in_features = load_backbone(name=cfg['model']['backbone'], pretrained=cfg['model']['pretrained_backbone'])
        
        for n, p in enumerate(self.parameters()):
            if n < cfg['model']['n_layers_freeze']:
                p.requires_grad_(False)

        for n, p in self.named_parameters():
            print('{} {}'.format(n, p.requires_grad))

        self.class_classifier = load_classifier(name=cfg['model']['class_classifier'], 
                                                input_size=in_features, 
                                                output_size=self.num_classes)

        self.domain_classifier = load_classifier(name=cfg['model']['domain_classifier'], 
                                                input_size=in_features, 
                                                output_size=2)

    def forward(self, x):
        lbda = self.get_lambda_p(self.get_p()) if self.mode == 'Train' else 0
        features = self.backbone(x)
        features = features.reshape(features.size(0), -1)
        reverse_features = GRL.apply(features, lbda)
        class_pred = self.class_classifier(features)
        domain_pred = self.domain_classifier(reverse_features)
        return class_pred, domain_pred

    def training_step(self, batch, batch_idx):
        # Apply Learning rate schedular
        self.lr_schedule_step(p=self.get_p())

        # Unpack source/target batch of images and labels
        (xs, ys), (xt, _) = batch

        # Generate fake labels for domains (0's for source and 1's for target)
        ys_domain = torch.zeros(self.cfg['training']['batch_size'], device=self.device, dtype=torch.long) 
        yt_domain = torch.ones(self.cfg['training']['batch_size'], device=self.device, dtype=torch.long) 

        # Predictions on source
        class_logit_src , domain_logit_src = self(xs)
        class_pred_src = F.softmax(class_logit_src, dim=1)

        loss_class_src = self.criterion_class(class_logit_src, ys)
        loss_domain_src = self.criterion_domain(domain_logit_src, ys_domain)

        # Predictions on target
        _, domain_logit_tgt = self(xt)
        loss_domain_tgt = self.criterion_domain(domain_logit_tgt, yt_domain)

        # Aggregate losses
        loss_domains = loss_domain_src + loss_domain_tgt
        loss_tot =  loss_class_src + loss_domains
        
        
        self.train_accuracy_class(class_pred_src, ys)
        self.train_accuracy_domain_src(torch.argmax(domain_logit_src, dim=1), ys_domain)
        self.train_accuracy_domain_tgt(torch.argmax(domain_logit_tgt, dim=1), yt_domain)

        self.log("Train/loss_tot", loss_tot)
        self.log("Train/loss_class_src", loss_class_src, prog_bar=True)
        self.log("Train/acc_class_src", self.train_accuracy_class, prog_bar=True)
        self.log("Train/acc_domain_src", self.train_accuracy_domain_src, prog_bar=True)
        self.log("Train/acc_domain_tgt", self.train_accuracy_domain_tgt, prog_bar=True)
        self.log("Train/loss_domains", loss_domains)
        return {"loss": loss_tot}

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        class_logit_src, _ = self(xs)
        class_pred_src = F.softmax(class_logit_src, dim=1)
        loss = self.criterion_class(class_logit_src, ys)
        return { 'loss': loss, 'preds': class_pred_src, 'targets': ys}

    def validation_step_end(self, outs):
        self.val_accuracy_class(outs["preds"], outs["targets"])
        self.log("Val/acc_src_step", self.val_accuracy_class, prog_bar=True)
        self.log("Val/loss_src_step", outs['loss'], prog_bar=True)


    def test_step(self, batch, batch_idx):
        (xs, ys), (xt, yt) = batch
        
        # Test Source
        class_logit_src, _ = self(xs)
        class_pred_src = F.softmax(class_logit_src, dim=1)
        loss_src = self.criterion_class(class_logit_src, ys)
        
        # Test Target
        class_logit_tgt, _ = self(xt)
        class_pred_tgt = F.softmax(class_logit_tgt, dim=1)
        loss_tgt = self.criterion_class(class_logit_tgt, yt)
        
        return {'loss_src': loss_src, 
                'loss_tgt': loss_tgt,
                'logits_src': class_logit_src, 
                'logits_tgt': class_logit_tgt, 
                'preds_src': class_pred_src, 
                'preds_tgt': class_pred_tgt,
                'labels_src': ys,
                'labels_tgt': yt}

    def test_step_end(self, outs):
        self.test_accuracy_class_src(outs["preds_src"], outs["labels_src"])
        self.test_accuracy_class_tgt(outs["preds_tgt"], outs["labels_tgt"])

        self.log("Test/acc_src_step", self.test_accuracy_class_src)
        self.log("Test/acc_tgt_step", self.test_accuracy_class_tgt)
        self.log("Test/loss_src_step", outs['loss_src'])
        self.log("Test/loss_tgt_step", outs['loss_tgt'])
        
        

    def configure_optimizers(self):
        model_parameter = [
            {
                "params": self.backbone.parameters(),
                "lr_mult": 0.1,
                'decay_mult': 2,
            },
            {
                "params": self.class_classifier.parameters(),
                "lr_mult": 1.0,
                'decay_mult': 2,
            },
            {
                "params": self.domain_classifier.parameters(),
                "lr_mult":  1.0,
                'decay_mult': 2,
            }
        ]

        if self.cfg['training']['optimizer']['type'] == "SGD":
            optimizer = torch.optim.SGD(
                model_parameter,
                lr=self.cfg['training']['optimizer']['lr'],
                momentum=self.cfg['training']['optimizer']['momentum'],
                weight_decay=self.cfg['training']['optimizer']['weight_decay'],
                nesterov=True)

        elif self.cfg['training']['optimizer']['type'] == "Adam":
            optimizer = torch.optim.Adam(
                model_parameter,
                lr=self.cfg['training']['optimizer']['lr'],
                betas=(self.cfg['training']['optimizer']['momentum'], 0.999),
                weight_decay=self.cfg['training']['optimizer']['weight_decay'])
        else:
             raise Exception("Optimizer not implemented yet, please use Adam or SGD.")
        return optimizer

    def lr_schedule_step(self, p):

        for param_group in self.optimizers().param_groups:
            
            # Update Learning rate
            param_group["lr"] = param_group["lr_mult"] * self.cfg['training']['optimizer']['lr'] / (1 + self.cfg['training']['scheduler']['alpha'] * p) ** self.cfg['training']['scheduler']['beta']
            
            # Update weight_decay
            param_group["weight_decay"] = self.cfg['training']['optimizer']['weight_decay'] * param_group["decay_mult"]

            # Logs Learning rate in wandb
            self.log("lr", param_group["lr"], on_epoch=False)

        # Logs Lambda in wandb
        self.log("lambda", self.get_lambda_p(self.get_p()), on_epoch=False)

    def get_p(self):
        current_iterations, current_epoch, len_dataloader = self.global_step, self.current_epoch, self.len_dataloader
        return float(current_iterations + current_epoch * len_dataloader) / self.cfg['training']['epochs'] / len_dataloader

    def get_lambda_p(self, p):
        return  2. / (1. + np.exp(-self.cfg['training']['scheduler']['gamma'] * p)) - 1

    def extract_features(self, x):
        features_backbone = self.backbone(x)
        features_backbone = features_backbone.reshape(features_backbone.size(0), -1)
        
        features_pretrained = self.backbone_fixed(x)
        features_pretrained = features_pretrained.reshape(features_pretrained.size(0), -1)
        return features_pretrained, features_backbone
