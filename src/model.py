"""
This module contains classes and functions relating to deep
neural network models whose parameters will be learned during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class CustomModel(nn.Module):
    def __init__(self):
        #TODO implement a (simple) CNN 
        raise NotImplemented()
    
    def forward(self, inputs):
        pass


def configure_transfer_learning_model(model_ft):
    """
    Given an existing model (possibly trained on an entirely different domain of images),
    change the final fully connected layer to suit the business-problem at hand (binary classification)
    """
    num_ftrs = model_ft.fc.in_features
    model_ft.fc = nn.Linear(num_ftrs, 2)
    return model_ft

def get_resnet18(pretrained=False):
    """
    Load the ResNet18 model, either just the structure or pretrained
    """
    Resnet18 = models.resnet18(pretrained=pretrained)
    Resnet18 = configure_transfer_learning_model(Resnet18)
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    Resnet18.to(device)
    
    return Resnet18
