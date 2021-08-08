# MADA-PL

This repository implements in Pytorch-Lightning:
  * (MADA) Multi-Adversarial Domain Adaptation (https://arxiv.org/abs/1809.02176) 
  * (DANN) Domain Adversarial Training (https://arxiv.org/abs/1505.07818) 

We also used the Weight and Biases (https://wandb.ai/) logger to track our experiments.

## Usage:

### Weights and Biases (Wandb)

To train our models we used [Weights and Biases](https://wandb.ai/site).
Weights and Biases allows to do experiment tracking, dataset versioning, model management, training visualisation for ML projects.
The code of this repository requires to create a [wand account](https://app.wandb.ai/login?signup=true) (for free using your github account) and to login using the [API KEY](https://wandb.ai/authorize).

### GPU

We trained our models on GPU, so our code requires to install NVIDIA Drivers.
Please check that your Local or Cloud machine is setup using:
```
nvidia-smi
```

### Training Configuration file
TODO


### Run Locally

```bash
pip3 install -r requirements.txt
```

```python
wandb login
```

```python
cd /Models/ && python3 /Models/run.py
```

### Run in a Docker Container (Recommended)

Install nvidia-docker2 to access GPU from a Docker container
Please visit: [Nvidia container-toolkit](https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html)

```bash
export EXPERIMENT_FOLDER= OWN_EXPERIMENT_FOLDER_PATH
```

```bash
docker build \
    --build-arg EXPERIMENT_PATH=$EXPERIMENT_FOLDER \
    -t train-mada .
```

```bash
docker run \
    -v $EXPERIMENT_FOLDER:$EXPERIMENT_FOLDER \
    --gpus all \
    -it \
    train-mada \
    bash
```

```python
root@28bbba0f0496:/ wandb login
```

```python
root@28bbba0f0496:/ cd /Models/
root@28bbba0f0496:/ python3 run.py --help # recommended

# Training
root@28bbba0f0496:/ python3 run.py --mode Train --cfg OWN_CONFIG.yml --experiment $EXPERIMENT_FOLDER

# Inference
root@28bbba0f0496:/ python3 run.py --mode Inference --cfg OWN_CONFIG.yml --ckpt MODEL_WEIGHTS_PATH --img_path IMAGE_PATH

```
