import torchvision
import torch.nn as nn


def decom_resnet50():
    resnet_net = torchvision.models.resnet50(
        weights=torchvision.models.ResNet50_Weights.DEFAULT
    )
    modules = list(resnet_net.children())[:-2]
    backbone = nn.Sequential(*modules)
    return backbone
