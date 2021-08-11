# MADA-PL

This repository implements in Pytorch-Lightning:
  * (DCNN) Deep Convolutional Neural Network using Transfer Learning.
  * (MADA) Multi-Adversarial Domain Adaptation (https://arxiv.org/abs/1809.02176).
  * (DANN) Domain Adversarial Training (https://arxiv.org/abs/1505.07818).

Pytorch-Lightning is a lightweight PyTorch wrapper for high-performance AI research. Please visit [Pytorch-Lightning documentation](https://pytorch-lightning.readthedocs.io/en/stable/) for more information.
We also used the [Weight and Biases](https://wandb.ai/) logger to track our experiments.

## Goal

The goal of (Multiple) Domain adversarial training is to train models so that they can perform better on images from other (targets) distributions compare to the distribution (source) on which they were trained using class labels. Using transfer learning and game therory, we can use unlabeled target image datasets to force the feature extractor (large pretrained convolution neural network) to generate background invariant features. Using adversarial training can help build more robust deep learning models.


## Datasets

Our code propose to train MADA or DANN using two famous datasets (like in the original papers):

 * [MNIST and MNISTM](https://paperswithcode.com/dataset/mnist-m) MNIST-M is created by combining MNIST digits with the patches randomly extracted from color photos of BSDS500 as their background. MNIST-M is often used for Domain adptation research.

 * [OFFICE31](https://paperswithcode.com/dataset/office-31) The Office dataset contains 31 object categories in three domains: Amazon, DSLR and Webcam. The 31 categories in the dataset consist of objects commonly encountered in office settings, such as keyboards, file cabinets, and laptops. The Amazon domain contains on average 90 images per class and 2817 images in total. As these images were captured from a website of online merchants, they are captured against clean background and at a unified scale. The DSLR domain contains 498 low-noise high resolution images (4288×2848). There are 5 objects per category. Each object was captured from different viewpoints on average 3 times. For Webcam, the 795 images of low resolution (640×480) exhibit significant noise and color as well as white balance artifacts. Office 31 is often used for Domain adptation research.

For both datasets, we splited the dataset into 3 subset: Train, Val, Test.
* For MNIST and MNIST, we have 50000 train images, 10000 validation images and 10000 Test Images
* For OFFICE31 we splited with the following ratio: 80% Train, 10% Val, 10% Test for each distributions (Amazon, DSLR and Webcam).

***Note:*** For MNIST and MNISTM, source and target dataset have the same size which make Batch spliting quite easy. However, for Office31, Amazon has much more data compare to DSLR or Webcam. Hence training on Office31 is more challenging. In our code, to simply the training each batch is define by taking the mininum between the source and target distribution.

***

## Usage:

### Weights and Biases (Wandb)

To train our models we used [Weights and Biases](https://wandb.ai/site).
Weights and Biases allows to do experiment tracking, dataset versioning, model management, training visualisation for ML projects.
The code of this repository requires to create a [wand account](https://app.wandb.ai/login?signup=true) (for free using your github account) and to login using the [API KEY](https://wandb.ai/authorize).
Weights and Biases will be install using the `requirements.txt`.

### Visualize Trainings experiments

Please check [our wandb runs dashboards](https://wandb.ai/marvtin/MADA-PL/runs).
Wandb allows to compare models performances and keep track of the past strategies.

### GPU

We trained our models on GPU, so our code requires to install NVIDIA Drivers.
Please check that your Local or Cloud machine is setup using:
```
nvidia-smi
```

### Training Configuration file

In this project, each model is define by a `yaml` configuration file. You can find examples here:
```bash
cat Models/config_DCNN.yml
cat Models/config_DANN.yml
cat Models/config_MADA.yml
```

You can choose many different things such as:
 * Source and Target Dataset. For MNIST classification, you use `MNIST`, `MNISTM`. For Office31 classification, you can use `AMAZON`, `DSLR`, `WEBCAM`.
 * Input image size (recommend 28 for MNIST, and 224 for Office31)
 * Transformation to apply on data (`transform_RGB_DA`, `transform_RGB`, `transform_mnist`, `transform_mnistm` ). Please look at `tools/utils.py` to find the one that best fit your desired transformation
 * Normalization (mean and std). If you use pretraine model, please use the correspond mean and std of the paper.
 * Model Type, that can either be `DANN` (Domain Aversarial Neural Network) or `MADA` (Multi-Adversarial Domain Adaptation)
 * Backbone Model(only RESNET) that can be: `resnet18`, `resnet34`, `resnet152`
 * Pretrained Weights from `imagenet` or just an empty model.
 * Number of layers to freeze (`0` means retrained the full backbone).
 * Fully-Connected Head classifier for both the class and domain. Can be either `linear2_dr2_bn`, `linear2_bn`, `linear3_bn2_v1`, `linear3_bn2_v2`.
 * Number of GPU and Workers (recommended 1 GPU and 0 Workers but depends on your machine)
 * Optimizer, which can be `Adam` or `SGD`. You can also choose the momentum.
 * Learning rate and Learning rate scheduler.
 * alpha, gamma, beta are hyperparameters that allows to compute the Lambda term (which goes from 0 to 1) in the Reversal Gradient Layer. According to both MADA and DANN paper `alpha=10`, `gamma=10`, `beta=0.75`. We recommend not to change them.
 * Batch size (The number of images per batch is difined by taking minimum between the source and target size)
 * Number of Epochs (recommend a large value for Office31).

To run your own training, you can modify directly this file but please make sure that the `key`:`value` component you changed are allowed by the code, otherwise you will get an error.


### Run Locally

- First install all the required libraries using `pip3`. We recommend to use a virtual environment.
```bash
pip3 install -r requirements.txt
```

- Next, login with your `wandb` account (look at your [API KEY](https://wandb.ai/authorize)).
```bash
wandb login
#or
python3 -m wandb login
```

-  Go to the Models folder and use the python file `run.py` to start a training or an inference. Please use `--help` before to get familiar with the command.
```bash
cd /Models/
python3 run.py --help # recommended
```

- Train your model based on your configuration file. You also need to provide a path where to store your model output (logs, weights, data). You can add `**\Experiment`, it will automatically create the folder `Experiment` for you. We recommend to use a `screen` before runing this command.
```bash
# Training
python3 run.py --mode Train --cfg OWN_CONFIG.yml --experiment EXPERIMENT_FOLDER_PATH
```
At this point, you should be able to track your training your `wandb` dashboard. Your dashboard can be found at https://wandb.ai/`YOUR_ACCOUNT`/MADA-PL. Here is our own [wandb Dashboard](https://wandb.ai/marvtin/MADA-PL).

- Once you obtain a good model, you can run an inference on a single input image using:
```bash
# Inference
python3 run.py --mode Inference --cfg OWN_CONFIG.yml --ckpt MODEL_WEIGHTS_PATH --img IMAGE_PATH
```


### Run from Docker Container

Install nvidia-docker2 to access GPU from a Docker container
Please visit: [Nvidia container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

- Create a shell variable `EXPERIMENT_FOLDER` that provide the experient path where to store your model output (logs, weights, data). You can add `**\Experiment`, it will automatically create the folder `Experiment` for you.

```bash
export EXPERIMENT_FOLDER= OWN_EXPERIMENT_FOLDER_PATH
```
***Note that OWN_EXPERIMENT_FOLDER_PATH must be an absolute path (avoid `~/` sortcut)**

- Build the nvidia docker image from the `Dockerfile`. Use the `--build-arg` to be able to create a volume for the experiment folder. In this example the image is called `train-mada`.
```bash
# From root of the repository (where Dockerfile is located)
docker build \
    --build-arg EXPERIMENT_PATH=$EXPERIMENT_FOLDER \
    -t  train-mada .
```

- Create a container from your image `train-mada`. Use `-it` flag to enter interactivaly your container. Use `-v` to create a volume between the host and container for the experiment folder.
```bash
docker run \
    -v $EXPERIMENT_FOLDER:$EXPERIMENT_FOLDER \
    --gpus all \
    -it \
    train-mada \
    bash
```

- Finally, just like locally, start a training (after wandb login) and/or run inferences.
```bash
root@28bbba0f0496:/ wandb login
root@28bbba0f0496:/ cd /Models/
root@28bbba0f0496:/ python3 run.py --help # recommended
# Train
root@28bbba0f0496:/ python3 run.py --mode Train --cfg OWN_CONFIG.yml --experiment $EXPERIMENT_FOLDER
# Inference
root@28bbba0f0496:/ python3 run.py --mode Inference --cfg OWN_CONFIG.yml --ckpt MODEL_WEIGHTS_PATH --img IMAGE_PATH
```

### Experiment Folder

After each training, a new folder is created in the experiment path you provided using `--experiment` flag.
The folder will be named as follow: `EXPERIMENT_FOLDER/MODELTYPE_DATASETSOURCE_TIMESTAMP`, for example, `DANN_AMAZON_1628056416`.
Inside, you will find your **configuration yaml** file, the **top 3 best models weights** saved during training (that minimize the validation source loss), and finally the **Models folder** that contains the source code.
This saving strategy allows to keep track of the experiments, compare configuration, and reuse the model using the original source code. 

### Notebook TSNE Exploration

Please visit `Notebooks/Latent-Space-Exploration.ipynb`. This notebook typically provide an example of how to run infrences without the run.py command. Also, We explored the latent space (resulting from the feature extract, e.g backbone) for all 3 models DCNN, DANN, MADA for the MNIST and MNISTM Dataset. The goal was to visualize in a 3D space the feature representation for each distribution. To reduce the dimensionality of the extracted features we used TSNE. Domain Adversarial models (DANN, MADA) are expected to have better feature representation on the target domain, and both distribution should be indistinguishable in the 3D space.

### Pretrained RESNET (Not retrained) Domain Distribution
![RESNET](/Notebooks/Backbone_pretrained_Feature_representation.png)

###  DCNN (Trained only on Source) Domain Distribution
![DCNN](/Notebooks/DCNN_Feature_representation.png)

### DANN (Trained using GRL) Domain Distribution
![DANN](/Notebooks/DANN_Feature_representation.png)

### MADA (Trained using GRL, class-wise) Domain Distribution
![MADA](/Notebooks/MADA_Feature_representation.png)

## Contributors
This project is part of a 2021 Deep Learning Class project (at Boston University CS523).
The code has been realized by Marvin Martin (marvtin@bu.edu) and Anirudh Mandahr (anirudh1@bu.edu).
We also thanks our Professor Peter Chin and Teacher Assistant Andrew Wood for supervising our work.



