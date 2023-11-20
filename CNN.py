import torch
from torch import nn
import torchvision
import torchvision.datasets
from torchvision import transforms
import os
import torch.utils.data as data
from PIL import Image

device = 'cuda' if torch.cuda.is_available else 'cpu'
# print(device)
BATCH_SIZE = 64
EPOCHS = 2
LEARNING_RATE = 0.0001
data_transform = transforms.Compose([
    transforms.ToTensor()
])
def run():
    train_data = torchvision.datasets.ImageFolder(root='Dataset', transform=data_transform)
    train_data_loader = data.DataLoader(train_data, BATCH_SIZE, True, num_workers=2)
    train_features_batch, train_labels_batch = next(iter(train_data_loader))
    class_names = train_data.classes
    class_dict = train_data.class_to_idx
    print(train_features_batch.shape)
    print(train_labels_batch.shape)

# print(f'Batch: {train_features_batch} Label: {train_labels_batch}')
if __name__ == '__main__':
    run()