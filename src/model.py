"""
This module contains classes and functions relating to deep
neural network models whose parameters will be learned during training
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models

class SimpleCNNModel(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.conv3 = nn.Conv2d(32, 64, 3, padding=1)
        
        self.fc1 = nn.Linear(64 * 4 * 4, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 1)
        
        #self.dropout = nn.Dropout(0.2)
        self.max_pool = nn.MaxPool2d(2,2)
        self.sigmoid = nn.Sigmoid()
    
    def forward(self, x):
        
        x = self.max_pool(F.relu(self.conv1(x)))
        x = self.max_pool(F.relu(self.conv2(x)))
        x = self.max_pool(F.relu(self.conv3(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = F.relu(self.fc1(x))
        #x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        
        return x

def configure_transfer_learning_model(model_ft):
    """
    Given an existing model (possibly trained on an entirely different domain of images),
    change the final fully connected layer to suit the business-problem at hand (binary classification)
    """
    num_ftrs = model_ft.fc.in_features
    #model_ft.fc = nn.Linear(num_ftrs, 2)
    model_ft.fc = nn.Sequential(
        nn.Linear(num_ftrs, 1),
        nn.Sigmoid(),
    )
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

def save_model(model, name):
    """
    Save the trained model to disk so that it can be re-loaded later
    :param model:    The trained instance of the model, whose weights will be preserved
    :param name:     A unique name for the model, will be used to save it disk and to identify it for reloading
    """
    torch.save(model.state_dict(), '{}.pt'.format(name))

def load_model(model, name):
    """
    Loads a previously trained model from disk
    :param model:    A new instance of the model class with untrained weights
    :param name:     The name of the model, will be used to load the appropriate file from disk
    """
    model.load_state_dict(torch.load('{}.pt'.format(name)))
    return model
