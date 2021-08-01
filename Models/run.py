import logging
import os
import glob
import datetime
import yaml
import torch
import wandb
from pathlib import Path
from PIL import Image, ImageFile
from shutil import copyfile, copytree, ignore_patterns
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import pytorch_lightning as pl

from tools.DataModule import DataModule
from tools.utils import ImagePredictionLogger
from tools.utils import get_transform
from nets.DANN import DANN
from nets.MADA import MADA

ImageFile.LOAD_TRUNCATED_IMAGES = True

class Runner:

    def __init__(self, mode='Inference', configfile_path=None, ckpt_path=None):

        # Check that configuration file exit
        assert (configfile_path != None), 'Error: Configuration file path not provided.'
        assert os.path.exists(configfile_path), 'Error: Configuration file path does not exist.'

        # If inference Check that ckpt_path file exit
        if mode == 'Inference':
            assert (ckpt_path != None), 'Error: Model weigths path not provided.'
            assert os.path.exists(ckpt_path), 'Error: Model weigths path does not exist.'


        # Load config file from yml
        with open(configfile_path) as f:
            self.cfg = yaml.load(f, Loader=yaml.FullLoader)

        # Attributes
        self.mode = mode
        self.configfile_path = configfile_path
        self.model_type = self.cfg['model']['type']
        self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        self.id = str(datetime.datetime.now().timestamp()).split('.')[0]


        if mode == 'Inference':

            if self.model_type == "DANN":
                self.net = DANN(self.cfg, mode) # create empty model
            if self.model_type == "MADA":
                self.net = MADA(self.cfg, mode) # create empty model

            self.net.to(self.device)
            self.net.load_state_dict(torch.load(ckpt_path, map_location=self.device)['state_dict'], strict=False) # load weights
            self.net.eval() # evaluation mode
            self.net.freeze() # evaluation mode
            self.transformation = self.cfg["input"]["dataset"]['transformation'] # keep the transformation name from cfg

        if mode == 'Train':

            # Setup seed for reproducibility
            pl.seed_everything(self.cfg['seed'])

            # Setup Experiment names
            experiment_root_dir = self.cfg['output']['save_path']
            dataset_name = self.cfg['input']['dataset']['name']
            run_name = f"{self.cfg['model']['type']}_{self.id}"
            self.exeriment_name = f'{self.model_type}_{dataset_name}_{self.id}'
            self.experiment_folder =  os.path.join(experiment_root_dir, self.exeriment_name, '')

            # Create Experiment folder
            try:
                if not os.path.exists(self.experiment_folder):
                    os.makedirs(self.experiment_folder)
            except OSError:
                print (f"Error: Failed creating experiment directory: {self.experiment_folder}")


            # Copy config file in the Experiment folder
            copyfile(configfile_path, os.path.join(self.experiment_folder, os.path.basename(os.path.normpath(configfile_path))))
            # Copy model code folder in the Experiment folder (ingnore wandb)
            copytree('MADA-PL/Models/', os.path.join(self.experiment_folder, 'model'), ignore=ignore_patterns('wandb'))


            # Wandb setup
            wandb.login()
            wandb.init(project="MADA-PL", name=run_name)

            # Copy in Artifact the model code source
            code = wandb.Artifact('project-source', type='code')
            for path in glob.glob('**/*.py', recursive=True):
                # SharkDetector already upladed automatically in init()
                if not 'run.py' in path:
                    code.add_file(path)
            wandb.run.use_artifact(code)


    def inference_predict(self, path_img):
        # check image path exists
        assert os.path.exists(path_img), 'Error: Image path not found.'

        # Check mode for Inference
        if self.mode == 'Inference':

            tr = get_transform(self.transformation) # retreive test transformation
            im = Image.open(path_img) # Load image from path
            imt = torch.unsqueeze(tr(im),0).to(self.device) # Transform and prepare image for the model

            # if DA
            if self.model_type == 'DANN':
                pred_class_logit, _ = self.net(imt) # predict logits
                return torch.sigmoid(pred_class_logit).item() #return final preds
        else:
            raise 'Cannot use inference_predict if not in Inference mode.'


    def train(self):

        print(f"\n######### {self.experiment_folder} #########\n")

        checkpoint_callback = ModelCheckpoint(monitor='Val/loss_step',
                                              dirpath=self.experiment_folder,
                                              filename= self.exeriment_name + '_epoch{epoch:02d}-val_loss_step{Val/loss_step:.2f}',
                                              auto_insert_metric_name=False,
                                              save_top_k=3,
                                              mode='min')

        self.dataModule = DataModule(self.cfg)
        self.dataModule.setup()
        samples_val_iter = iter(self.dataModule.val_dataloader()) # Source Data
        samples_test_iter = iter(self.dataModule.test_dataloader())
        callbacks = [checkpoint_callback,
                    ImagePredictionLogger(samples_val_iter, title="Val_Preds"),
                    ImagePredictionLogger(samples_test_iter,  title="Test_Preds")]

        if self.model_type == "DANN":
            self.net = DANN(self.cfg,
                            mode='Train',
                            dataModule=self.dataModule)

        if self.model_type == "MADA":
            self.net = MADA(self.cfg,
                            mode='Train',
                            dataModule=self.dataModule)


        wandb_logger = WandbLogger(project= "MADA-PL",
                                   name=f"{self.model_type}_{self.id}")


        self.trainer = pl.Trainer(
                        deterministic=True,
                        val_check_interval=0.25,
                        gpus=self.cfg['training']['gpus'],
                        logger=wandb_logger,
                        log_every_n_steps=10,
                        callbacks=callbacks,
                        max_epochs=self.cfg['training']['epochs'],
                        progress_bar_refresh_rate=25)

        self.trainer.fit(self.net, self.dataModule)
        print(f"\n######### Best model path is : {checkpoint_callback.best_model_path} #########\n")

        if self.cfg['output']['save']:
            for exp_path in os.listdir(self.experiment_folder):
                if exp_path != 'model':
                    file_path =  os.path.join(self.experiment_folder, exp_path)
                    print("Saving to Wanbd: " + exp_path)
                    wandb.save(os.path.join(wandb.run.dir, file_path))

            ## just in case save last
            model_filename_pth = os.path.join(self.experiment_folder, f"{self.exeriment_name}_last_last.ckpt")
            self.trainer.save_checkpoint(model_filename_pth)
            wandb.save(os.path.join(wandb.run.dir, model_filename_pth))


    def test(self):
        self.dataModule.setup(stage='test')
        self.trainer.test(ckpt_path='best')
        wandb.finish()


if __name__ == '__main__':

    runner = Runner(mode='Train', configfile_path="config.yml")
    runner.train()
    runner.test()


    #runner = Runner(
    #                mode='Inference',
    #                configfile_path="config_DANN.yml",
    #                ckpt_path='....ckpt')
    #pred = runner.inference_predict('....png')
    #print(pred)
