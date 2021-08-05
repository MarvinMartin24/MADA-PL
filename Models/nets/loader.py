import torchvision
from collections import OrderedDict
import torch.nn as nn


def load_backbone(name=None, pretrained=False):
    if name == "alexnet":
        print('Load resnet18, pretrained weights : {}'.format(pretrained))
        alexnet = torchvision.models.alexnet(pretrained=True if pretrained == 'imagenet' else False)
        # remove fully connected layer:
        features = nn.Sequential(*list(resnet.children())[:-1])
        in_features = alexnet.classifier[1].in_features

    if name == "resnet18":
        print('Load resnet18, pretrained weights : {}'.format(pretrained))
        resnet = torchvision.models.resnet18(pretrained=True if pretrained == 'imagenet' else False)
        # remove fully connected layer:
        features = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features

    if name == "resnet34":
        print('Load resnet34, pretrained weights : {}'.format(pretrained))
        resnet = torchvision.models.resnet34(pretrained=True if pretrained == 'imagenet' else False)
        # remove fully connected layer:
        features = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features

    if name == "resnet152":
        print('Load resnet152, pretrained weights : {}'.format(pretrained))
        resnet = torchvision.models.resnet152(pretrained=True if pretrained == 'imagenet' else False)
        # remove fully connected layer:
        features = nn.Sequential(*list(resnet.children())[:-1])
        in_features = resnet.fc.in_features

    return features, in_features

def load_classifier(name, input_size=None, output_size=None):
    if name == 'linear2_dr2_bn':
        return linear2_dr2_bn(input_size, output_size)
    elif name == 'linear2_bn':
        return linear2_bn(input_size, output_size)
    elif name == 'linear3_bn2_v1':
        return linear3_bn2_v1(input_size, output_size)
    elif name == 'linear3_bn2_v2':
        return linear3_bn2_v2(input_size, output_size)
    raise Exception('Name of classifier network given in config not correct')
    
def linear3_bn2_v1(input_size, output_size):
    return nn.Sequential(OrderedDict([
                                        ('d1', nn.Linear(input_size, 3072)),
                                        ('bn1', nn.BatchNorm1d(3072)),
                                        ('relu1', nn.ReLU()),
                                        ('d2', nn.Linear(3072, 2048)),
                                        ('bn2', nn.BatchNorm1d(2048)),
                                        ('relu2', nn.ReLU()),
                                        ('d3', nn.Linear(2048, output_size))
                                     ]))

def linear3_bn2_v2(input_size, output_size):
    return nn.Sequential(OrderedDict([
                                        ('d1', nn.Linear(input_size, 1024)),
                                        ('bn1', nn.BatchNorm1d(1024)),
                                        ('relu1', nn.ReLU()),
                                        ('d2', nn.Linear(1024, 1024)),
                                        ('bn2', nn.BatchNorm1d(1024)),
                                        ('relu2', nn.ReLU()),
                                        ('d3', nn.Linear(1024, output_size))
                                     ]))
    

def linear2_dr2_bn(input_size, output_size):
    return nn.Sequential(OrderedDict([
                                        ('dr1', nn.Dropout(0.5)),
                                        ('d1', nn.Linear(input_size, 2048)),
                                        ('bn1', nn.BatchNorm1d(2048)),
                                        ('relu1', nn.ReLU()),
                                        ('dr2', nn.Dropout(0.5)),
                                        ('d2', nn.Linear(2048, output_size))
                                     ]))

def linear2_bn(input_size, output_size):
    return nn.Sequential(OrderedDict([
                                        ('d1', nn.Linear(input_size, 100)),
                                        ('bn1',nn.BatchNorm1d(100)),
                                        ('relu1',nn.ReLU(True)),
                                        ('d2',nn.Linear(100, output_size))
                                    ]))