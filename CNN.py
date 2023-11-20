import torch
from torch import nn
import torchvision
import torchvision.datasets
import os
import random
# os.remove("demofile.txt")

# print(torch.__version__, torchvision.__version__)
# device = 'cuda' if torch.cuda.is_available else 'cpu'
#
# print(device)
#
# root = torchvision.datasets.ImageFolder(root='Dataset')
#
# print(root)

# files = os.listdir('Dataset/NoChange')
# to_delete = []

# def get_rnd(max):
#     rnd = random.randint(1, max)
#     return rnd
#
# for i in range(len(files)-4993):
#     while True:
#         rnd = get_rnd(len(files))
#         if to_delete.__contains__(rnd):
#             pass
#         else:
#             to_delete.append(rnd)
#             break
#
#
# to_delete.sort(reverse=True)
# print(f'{len(to_delete)} and {len(files)-4993}')
# print(to_delete)


# for x in to_delete:
#     if os.path.exists(f"Dataset/NoChange/{files[x-1]}"):
#         os.remove(f"Dataset/NoChange/{files[x-1]}")
#     else:
#         print("The file does not exist")