import random

import torch
from torch import nn
import torchvision
import torchvision.datasets
from torchvision import transforms
import os
import shutil
import torch.utils.data as data
from PIL import Image
import glob


device = 'cuda' if torch.cuda.is_available else 'cpu'
# print(device)
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 0.0001
data_transform = transforms.Compose([
    transforms.ToTensor()
])
def run():
    train_data = torchvision.datasets.ImageFolder(root='Dataset/Train', transform=data_transform)
    train_data_loader = data.DataLoader(train_data, BATCH_SIZE, True, num_workers=2)
    test_data = torchvision.datasets.ImageFolder(root='Dataset/Test', transform=data_transform)
    test_data_loader = data.DataLoader(test_data, BATCH_SIZE, True, num_workers=2)
    train_features_batch, train_labels_batch = next(iter(train_data_loader))
    class_names = train_data.classes
    class_dict = train_data.class_to_idx
    # print(train_features_batch)
    # print(train_labels_batch)
    # img, label = train_data[5000][0], train_data[5000][1]
    # print(train_data.classes[label])
    print(train_data_loader.dataset.__len__())
    print(test_data_loader.dataset.__len__())

# print(f'Batch: {train_features_batch} Label: {train_labels_batch}')
if __name__ == '__main__':
    run()
