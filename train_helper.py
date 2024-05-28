import os

import torch
from torchvision import transforms
from torchvision.transforms import Compose

from dataset import BoneFractureDataset
from utils import *

# setting device to 'mps' for mac

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

# setting current directory path to dataset

dataset_path = os.path.join(os.getcwd(), 'Dataset')

# creating train, test, and validation dataset paths

train_dir = os.path.join(dataset_path, 'train')
val_dir = os.path.join(dataset_path, 'val')
test_dir = os.path.join(dataset_path, 'test')

# creating list of image paths in each directory

train_image_paths = create_files(train_dir)
val_image_paths = create_files(val_dir)
test_image_paths = create_files(test_dir)

# setting a small subset of datasets for testing

train_image_paths = train_image_paths[:200]
val_image_paths = val_image_paths[:96]
test_image_paths = test_image_paths[:64]

# composing transformations to be applied to our images

transform = Compose([
        transforms.ToPILImage(),
        transforms.Resize(224),
        transforms.CenterCrop(224),
        transforms.RandomHorizontalFlip(p = 0.5),
        transforms.ToTensor(),
        transforms.Normalize(mean = [0.485, 0.456, 0.406],
                             std = [0.229, 0.224, 0.225] )
        ])

# initialising train, test, and validation datasets

train_dataset = BoneFractureDataset(train_image_paths, transform)
val_dataset = BoneFractureDataset(val_image_paths, transform)
test_dataset = BoneFractureDataset(test_image_paths, transform)

# setting up batch size for our datasets

batch_size = 16

# creating dataloaders for our datasets

train_data = create_data(train_dataset, batch_size)
val_data = create_data(val_dataset, batch_size)
test_data = create_data(test_dataset, batch_size)