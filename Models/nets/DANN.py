import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict
import wandb
import os

from nets.GRL import GRL
from nets.backbones import load_backbone


class DANN(pl.LightningModule):

    def __init__(self, cfg, mode, dataModule=None):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        if mode == 'Train':

          self.len_dataloader = dataModule.len_dataloader
          self.criterion_classif = nn.CrossEntropyLoss()
          self.criterion_domain = F.cross_entropy

          # log hyperparameters
          self.save_hyperparameters()

          # compute the accuracy
          self.train_accuracy_classif =  pl.metrics.Accuracy()
          self.train_accuracy_domain_src =  pl.metrics.Accuracy()
          self.train_accuracy_domain_tgt =  pl.metrics.Accuracy()
          self.val_accuracy_classif =  pl.metrics.Accuracy()
          self.test_accuracy_classif =  pl.metrics.Accuracy()


        self.backbone, in_features = load_backbone(name=cfg['model']['backbone'], pretrained=cfg['model']['pretrained_backbone'])

        for n, p in enumerate(self.parameters()):
            if n < cfg['model']['n_layers_freeze']:
                p.requires_grad_(False)

        for n, p in self.named_parameters():
            print('{} {}'.format(n, p.requires_grad))

        self.class_classifier = nn.Sequential(
                                    OrderedDict([
                                        ('d1', nn.Linear(in_features, 2048)),
                                        ('bn1', nn.BatchNorm1d(2048)),
                                        ('relu1', nn.ReLU()),
                                        ('dr2', nn.Dropout(0.5)),
                                        ('d2', nn.Linear(2048, dataModule.num_classes))
                                    ]))

        self.domain_classifier = nn.Sequential(
                                    OrderedDict([
                                        ('d1', nn.Linear(in_features, 100)),
                                        ('bn1',nn.BatchNorm1d(100)),
                                        ('relu1',nn.ReLU(True)),
                                        ('d2',nn.Linear(100, 2))
                                    ]))

    def forward(self, x):
        lbda = self.get_lambda_p(self.get_p()) if self.mode == 'Train' else 0
        features = self.backbone(x)
        features = features.reshape(features.size(0), -1)
        reverse_features = GRL.apply(features, lbda)
        class_pred = self.class_classifier(features)
        domain_pred = self.domain_classifier(reverse_features)
        return class_pred, domain_pred

    def training_step(self, batch, batch_idx):

        self.lr_schedule_step(p=self.get_p())

        (xs, ys), (xt, _) = batch

        ys_domain = torch.zeros(self.cfg['training']['batch_size'], device=self.device, dtype=torch.long) # generate source domain labels
        yt_domain = torch.ones(self.cfg['training']['batch_size'], device=self.device, dtype=torch.long) # generate target domain labels

        # on source domain
        class_src_pred , domain_src_pred = self(xs)

        loss_classif_src = self.criterion_classif(class_src_pred, ys)
        loss_domain_src = self.criterion_domain(domain_src_pred, ys_domain)

        # on target domain
        _, domain_tgt_pred = self(xt)
        loss_domain_tgt = self.criterion_domain(domain_tgt_pred, yt_domain)

        # Aggregate loss
        loss_domain = loss_domain_src + loss_domain_tgt
        loss_tot =  loss_classif_src + loss_domain
        
        
        self.train_accuracy_classif(F.softmax(class_src_pred, dim=1), ys)
        self.train_accuracy_domain_src(torch.argmax(domain_src_pred, dim=1), ys_domain)
        self.train_accuracy_domain_tgt(torch.argmax(domain_tgt_pred, dim=1), yt_domain)

        self.log("Train/loss_tot", loss_tot)
        self.log("Train/loss_classif", loss_classif_src, prog_bar=True)
        self.log("Train/loss_domain", loss_domain)
        self.log("Train/acc_classif", self.train_accuracy_classif, prog_bar=True)
        self.log("Train/acc_domain_src", self.train_accuracy_domain_src)
        self.log("Train/acc_domain_tgt", self.train_accuracy_domain_tgt)

        return {"loss": loss_tot, 'log_loss': { 'classif':loss_classif_src,  'domain': loss_domain}}


    def training_epoch_end(self, outs):
        self.log("Train/acc_classif_epoch", self.train_accuracy_classif.compute(), prog_bar=True)
        self.log("Train/acc_domain_src_epoch", self.train_accuracy_domain_src.compute(), prog_bar=True)
        self.log("Train/acc_domain_tgt_epoch", self.train_accuracy_domain_tgt.compute(), prog_bar=True)

        self.log('Train/loss_tot_epoch', torch.stack([x["loss"] for x in outs]).mean())
        self.log("Train/loss_classif_epoch", torch.stack([x["log_loss"]["classif"] for x in outs]).mean())
        self.log("Train/loss_domain_epoch", torch.stack([x["log_loss"]["domain"] for x in outs]).mean())

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        class_src_pred, _ = self(xs)
        loss = self.criterion_classif(class_src_pred, ys)
        return { 'loss': loss, 'logits': class_src_pred, 'preds': F.softmax(class_src_pred, dim=1), 'targets': ys}

    def validation_step_end(self, outs):
        self.val_accuracy_classif(outs["preds"], outs["targets"])
        self.log("Val/acc_step", self.val_accuracy_classif, prog_bar=True)
        self.log("Val/loss_step", outs['loss'], prog_bar=True)
        return outs

    def validation_epoch_end(self, outs):
        flattened_logits = torch.flatten(torch.cat([outs[0]["logits"]]))
        self.logger.experiment.log({
            "Val/logits_epoch": wandb.Histogram(flattened_logits.to("cpu")),
            "Val/loss_epoch":  torch.stack([x["loss"] for x in outs]).mean(),
            "Val/acc_epoch":  self.val_accuracy_classif.compute(),
        })

    def test_step(self, batch, batch_idx):
        xs, ys = batch
        class_src_pred, _ = self(xs)
        loss = self.criterion_classif(class_src_pred, ys)
        return { 'loss': loss, 'logits': class_src_pred, 'preds': F.softmax(class_src_pred, dim=1), 'targets': ys}

    def test_step_end(self, outs):
        self.test_accuracy_classif(outs["preds"], outs["targets"])
        self.log("Test/acc_step", self.test_accuracy_classif)
        self.log("Test/loss_step", outs['loss'])

    def test_epoch_end(self, outs):
        self.logger.experiment.log({
            "Test/acc_epoch":  self.test_accuracy_classif.compute()
        })

    def test_end(self, outs):
        avg_loss = torch.stack([x['loss'] for x in outs]).mean()
        avg_acc = test_accuracy_classif.compute()
        logs = {'Test/test_loss': avg_loss, 'Test/test_acc': avg_acc}
        self.log('Test/test_loss_end', avg_loss)
        self.log('Test/test_acc_end', avg_acc)
        return {'Test/loss_end': avg_loss, 'Test/acc_end': avg_acc, 'log': logs, 'progress_bar': logs}

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

        if self.cfg['training']['optimizer']['type'] == "Adam":
            optimizer = torch.optim.Adam(
                model_parameter,
                lr=self.cfg['training']['optimizer']['lr'],
                betas=(0.9, 0.999),
                weight_decay=self.cfg['training']['optimizer']['weight_decay'])
        else:
             raise("Optimizer not implemented yet, please use Adam or SGD.")
        return optimizer

    def lr_schedule_step(self, p):

        for param_group in self.optimizers().param_groups:
            # Update Learning rate
            param_group["lr"] = param_group["lr_mult"] * self.cfg['training']['optimizer']['lr'] / (1 + self.cfg['training']['scheduler']['alpha'] * p) ** self.cfg['training']['scheduler']['beta']
            # Update weight_decay
            param_group["weight_decay"] = self.cfg['training']['optimizer']['weight_decay'] * param_group["decay_mult"]

            # Logs
            self.log("lr", param_group["lr"], on_epoch=True)
            self.log("weight_decay", param_group["weight_decay"], on_epoch=True)

        # Logs
        self.log("lambda", self.get_lambda_p(self.get_p()), on_epoch=True)
        self.log("p", self.get_p(), on_epoch=True)

    def get_p(self):
        current_iterations, current_epoch, len_dataloader = self.global_step, self.current_epoch, self.len_dataloader
        return float(current_iterations + current_epoch * len_dataloader) / self.cfg['training']['epochs'] / len_dataloader

    def get_lambda_p(self, p):
        return  2. / (1. + np.exp(-self.cfg['training']['scheduler']['gamma'] * p)) - 1
