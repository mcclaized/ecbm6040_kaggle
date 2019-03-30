"""
This module contains classes and functions relating to loading a pre-processing
the training data to make it suitable for input to a PyTorch model
"""

import os
import time
import shutil

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import ImageFolder

def get_image_datasets(data_dir, transforms=None):
    """
    The ImageFolder dataset class expects to see:
    root_dir/class1/image1.png
    root_dir/class2/image3.png
    
    So in our case, the function train_test_split creates a structure like:
    train/0/0036b31b8c5420f8c640a11cd3f8a375f4c56256.tif
    
    This dataset then yields a tuple of (image, label)
    """
    datasets = {}
    for phase in transforms:
        datasets[phase] = ImageFolder(
            os.path.join(data_dir, phase),
            transforms[phase],
        )
    return datasets

def get_image_dataloaders(data_dir, transforms=None, **params):
    """
    Construct a DataLoader for each of the image datasets that will provide batches
    of data to the model to train
    """
    dataloaders = {}
    for phase, dataset in get_image_datasets(data_dir, transforms=transforms).items():
        dataloaders[phase] = DataLoader(dataset, **params)
    return dataloaders

def display_image_batch(inp, title=None):
    """Display an image given as a Tensor"""
    
    # swap color axis because
    # numpy image: H x W x C
    # torch image: C X H X W
    inp = inp.numpy().transpose((1, 2, 0))
    
    ## Add back in if we do a Normalize transform
    # mean = np.array([0.485, 0.456, 0.406])  
    # std = np.array([0.229, 0.224, 0.225])
    # inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)
    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def train_test_split(train_labels_csv, data_root_dir, train_split=0.8):
    """
    This function takes the training data in the sub-directory 'train_all' in the data_root_dir
    and splits them into training data and validation data by the provided train_split proportion.
    
    This is done by copying files on disk into the new directories 'train' and 'validate',
    so as to interop with existing PyTorch DataLoaders
    
    WARNING: This function has side-effects!
    
    :param train_labels_csv:    The path to the CSV file that has the name and label for each training example
    :param data_root_dir:       The directory that contains a sub-dir 'train_all' and is where new directories will be made
    :train_split:               The proportion of training examples that should be used for training. 
                                The remainder will be used for validation.
    """
    for split_dir in ("train", "validate"):
        if os.path.isdir(os.path.join(data_root_dir, split_dir)):
            print("Training directory '{}' already exists, recreating it in 5 seconds...".format(split_dir))
            time.sleep(5)
            shutil.rmtree(os.path.join(data_root_dir, split_dir))
        os.makedirs(os.path.join(data_root_dir, split_dir, "0"))
        os.makedirs(os.path.join(data_root_dir, split_dir, "1"))

    train_labels = pd.read_csv(train_labels_csv)
    # randomly assign each image to the training set with probability 'train_split'
    train = np.random.binomial(1, train_split, train_labels.shape[0])

    print("Splitting the files from train_all/ into train/ and validate/ ...")
    for (idx, image_file, label) in train_labels.itertuples():
        src = os.path.join(
            data_root_dir, "train_all", "{}.tif".format(image_file)
        )
        dst = os.path.join(
            data_root_dir, "train" if train[idx] else "validate", str(label), "{}.tif".format(image_file)
        )
        shutil.copyfile(src, dst)
        if idx and not idx % 50000:
            print("Successfully copied {} files".format(idx))
    print("Finished splitting the input files!")
