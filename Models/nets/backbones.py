import torchvision
import torch.nn as nn


def load_backbone(name=None, pretrained=False):
    """
    Function that load backbone from torchvision
    """

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
