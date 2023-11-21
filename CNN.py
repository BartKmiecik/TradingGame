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


# device = 'cuda' if torch.cuda.is_available else 'cpu'
# # print(device)
# BATCH_SIZE = 64
# EPOCHS = 2
# LEARNING_RATE = 0.0001
# data_transform = transforms.Compose([
#     transforms.ToTensor()
# ])
# def run():
#     train_data = torchvision.datasets.ImageFolder(root='Dataset', transform=data_transform)
#     train_data_loader = data.DataLoader(train_data, BATCH_SIZE, True, num_workers=2)
#     train_features_batch, train_labels_batch = next(iter(train_data_loader))
#     class_names = train_data.classes
#     class_dict = train_data.class_to_idx
#     print(train_features_batch.shape)
#     print(train_labels_batch.shape)
#
# # print(f'Batch: {train_features_batch} Label: {train_labels_batch}')
# if __name__ == '__main__':
#     run()

# shutil.move("path/to/current/file.foo", "path/to/new/destination/for/file.foo")
all_files = glob.glob("Dataset/Train/Rise/*.png")
print(len(all_files))

def get_rand(max):
    return random.randint(0, max-1)

to_move = []
for i in range(int(.2 * len(all_files))):
    rnd = get_rand(len(all_files))
    while to_move.__contains__(rnd):
        rnd = get_rand(len(all_files))
    to_move.append(rnd)

# print(f'{len(to_move)} and len of max: {.2 * len(all_files)}')

for x in to_move:
    print(os.path.basename(all_files[x]))
    os.replace(f"Dataset/Train/Rise/{os.path.basename(all_files[x])}", f"Dataset/Test/Rise/{os.path.basename(all_files[x])}")