import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
import pytorch_lightning as pl
from collections import OrderedDict
import wandb
import os

from nets.GRL import GRL
from nets.loader import load_backbone


class MADA(pl.LightningModule):

    def __init__(self, cfg, mode, dataModule=None):
        super().__init__()
        self.cfg = cfg
        self.mode = mode

        if mode == 'Train':
            
            self.len_dataloader = dataModule.len_dataloader
            self.criterion_class = nn.CrossEntropyLoss()
            self.criterion_domain = nn.CrossEntropyLoss()
            
            # log hyperparameters
            self.save_hyperparameters()

            # Init Train metrics for source and targets
            self.train_accuracy_class =  pl.metrics.Accuracy() # 1 source class accuarcies
            self.train_accuracy_domain_src =  pl.metrics.Accuracy() # 1 source domain accuarcies
            self.train_accuracy_domain_tgts = {} # K targets domain accuarcies
            for tqt in self.cfg['input']['dataset']['tgts']:
                self.train_accuracy_domain_tgts[f"train_accuracy_domain_tgt_{tqt}"] = pl.metrics.Accuracy()
            
            # Init Val metrics for source (only on the class)
            self.val_accuracy_class =  pl.metrics.Accuracy()
            
            # Init test metrics for source and targets
            self.test_accuracy_class_src =  pl.metrics.Accuracy() # 1 source class accuarcies
            self.test_accuracy_class_tqts = {} # K targets class accuarcies
            for tqt in self.cfg['input']['dataset']['tgts']:
                self.test_accuracy_class_tqts[f"test_accuracy_class_tqt_{tqt}"] = pl.metrics.Accuracy()
                

        self.backbone, in_features = load_backbone(name=cfg['model']['backbone'], pretrained=cfg['model']['pretrained_backbone'])

        for n, p in enumerate(self.parameters()):
            if n < cfg['model']['n_layers_freeze']:
                p.requires_grad_(False)

        #for n, p in self.named_parameters():
            #print('{} {}'.format(n, p.requires_grad))

        self.class_classifier = load_classifier(name=cfg['model']['class_classifier'], 
                                                input_size=in_features, 
                                                output_size=dataModule.num_classes)

        self.domain_classifiers = {}
        for tqt in self.cfg['input']['dataset']['tgts']:
            self.domain_classifiers[f"domain_classifier_{tqt}"] = load_classifier(name=cfg['model']['domain_classifier'], 
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

        self.lr_schedule_step(p=self.get_p())

        (xs, ys), (xt, _) = batch

        ys_domain = torch.zeros(self.cfg['training']['batch_size'], device=self.device, dtype=torch.long) # generate source domain labels
        yt_domain = torch.ones(self.cfg['training']['batch_size'], device=self.device, dtype=torch.long) # generate target domain labels

        # on source domain
        class_src_pred , domain_src_pred = self(xs)

        loss_class_src = self.criterion_class(class_src_pred, ys)
        loss_domain_src = self.criterion_domain(domain_src_pred, ys_domain)

        # on target domain
        _, domain_tgt_pred = self(xt)
        loss_domain_tgt = self.criterion_domain(domain_tgt_pred, yt_domain)

        # Aggregate loss
        loss_domain = loss_domain_src + loss_domain_tgt
        loss_tot =  loss_class_src + loss_domain
        
        
        self.train_accuracy_class(F.softmax(class_src_pred, dim=1), ys)
        self.train_accuracy_domain_src(torch.argmax(domain_src_pred, dim=1), ys_domain)
        self.train_accuracy_domain_tgt(torch.argmax(domain_tgt_pred, dim=1), yt_domain)

        self.log("Train/loss_tot", loss_tot)
        self.log("Train/loss_class", loss_class_src, prog_bar=True)
        self.log("Train/loss_domain", loss_domain)
        self.log("Train/acc_class", self.train_accuracy_class, prog_bar=True)
        self.log("Train/acc_domain_src", self.train_accuracy_domain_src)
        self.log("Train/acc_domain_tgt", self.train_accuracy_domain_tgt)

        return {"loss": loss_tot, 'log_loss': { 'class':loss_class_src,  'domain': loss_domain}}


    def training_epoch_end(self, outs):
        self.log("Train/acc_class_epoch", self.train_accuracy_class.compute(), prog_bar=True)
        self.log("Train/acc_domain_src_epoch", self.train_accuracy_domain_src.compute(), prog_bar=True)
        self.log("Train/acc_domain_tgt_epoch", self.train_accuracy_domain_tgt.compute(), prog_bar=True)

        self.log('Train/loss_tot_epoch', torch.stack([x["loss"] for x in outs]).mean())
        self.log("Train/loss_class_epoch", torch.stack([x["log_loss"]["class"] for x in outs]).mean())
        self.log("Train/loss_domain_epoch", torch.stack([x["log_loss"]["domain"] for x in outs]).mean())

    def validation_step(self, batch, batch_idx):
        xs, ys = batch
        class_src_pred, _ = self(xs)
        loss = self.criterion_class(class_src_pred, ys)
        return { 'loss': loss, 'logits': class_src_pred, 'preds': F.softmax(class_src_pred, dim=1), 'targets': ys}

    def validation_step_end(self, outs):
        self.val_accuracy_class(outs["preds"], outs["targets"])
        self.log("Val/acc_step", self.val_accuracy_class, prog_bar=True)
        self.log("Val/loss_step", outs['loss'], prog_bar=True)
        return outs


    def test_step(self, batch, batch_idx):
        # Lists to store targets logits, predictions, losses, labels
        class_logits_tgts, class_preds_tgts, losses_tqts, yts = [], [], [], []
        
        # Unpack each testset for all domains
        (xs, ys), batch_tqts = batch[0], batch[1:]
        
        # Prediction of the source testset  
        class_logit_src, _ = self(xs)
        class_pred_src = F.softmax(class_logit_src, dim=1)
        
        # Compute the loss of the source testset  
        loss_src = self.criterion_class(class_logit_src, ys)
        
        # For each target testset
        for (xt, yt) in batch_tqts:
            
            # Prediction of the source testset  
            class_logit_tqt, _ = self(xt)
            yts.append(yt)
            class_pred_tqt = F.softmax(class_logit_tqt, dim=1)
            class_logits_tgts.append(class_logit_tqt)
            class_preds_tgts.append(class_pred_tqt)
            
            # Compute the loss of the target testset  
            loss_tqt = self.criterion_class(class_logit_tqt, yt)
            losses_tqts.append(loss_tqt)
        
        return {'loss_src': loss_src, 
                'losses_tqts': losses_tqts,
                'logits_src': class_logit_src, 
                'preds_src': class_pred_src, 
                'logits_tqts': class_logits_tgts,
                'preds_tqts': class_preds_tgts,
                'targets_src': ys,
                'targets_tqts': yts}

    def test_step_end(self, outs):
        self.test_accuracy_class(outs["preds"], outs["targets"])
        
        self.log("Test/acc_step", self.test_accuracy_class)
        self.log("Test/loss_step", outs['loss'])


    def test_end(self, outs):
        avg_loss = torch.stack([x['loss'] for x in outs]).mean()
        avg_acc = test_accuracy_class.compute()
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
             raise Exception("Optimizer not implemented yet, please use Adam or SGD.")
        return optimizer

    def lr_schedule_step(self, p):

        for param_group in self.optimizers().param_groups:
            # Update Learning rate
            param_group["lr"] = param_group["lr_mult"] * self.cfg['training']['optimizer']['lr'] / (1 + self.cfg['training']['scheduler']['alpha'] * p) ** self.cfg['training']['scheduler']['beta']
            # Update weight_decay
            param_group["weight_decay"] = self.cfg['training']['optimizer']['weight_decay'] * param_group["decay_mult"]
            # Logs
            self.log("lr", param_group["lr"], on_epoch=False)
        # Logs
        self.log("lambda", self.get_lambda_p(self.get_p()), on_epoch=False)

    def get_p(self):
        current_iterations, current_epoch, len_dataloader = self.global_step, self.current_epoch, self.len_dataloader
        return float(current_iterations + current_epoch * len_dataloader) / self.cfg['training']['epochs'] / len_dataloader

    def get_lambda_p(self, p):
        return  2. / (1. + np.exp(-self.cfg['training']['scheduler']['gamma'] * p)) - 1
