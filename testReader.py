import torchvision
import torchvision.datasets as dset
import torchvision.transforms as transforms
from torch.utils.data import DataLoader,Dataset
import matplotlib.pyplot as plt
import torchvision.utils
import numpy as np
import random
from PIL import Image
import torch
from torch.autograd import Variable
import PIL.ImageOps    
import torch.nn as nn
from torch import optim
import torch.nn.functional as F
import os
root_path = os.getcwd()
import sys
sys.path.append(root_path)

from config import Config
from utils import imshow
from data_reader import SiameseNetworkDataset

folder_dataset = dset.ImageFolder(root=Config.training_dir)
siamese_dataset = SiameseNetworkDataset(
  imageFolderDataset=folder_dataset,
  transform=transforms.Compose(
    [
      transforms.Resize((100,100)),
      transforms.ToTensor()
    ]
  ),
  should_invert=False)
for i in range(5):
  print(siamese_dataset.imageFolderDataset.imgs[i])

vis_dataloader = DataLoader(siamese_dataset,
                        shuffle=True,
                        num_workers=8,
                        batch_size=8)
dataiter = iter(vis_dataloader)
example_batch = next(dataiter)
print(example_batch[0].shape)
concatenated = torch.cat((example_batch[0],example_batch[1]),0)
imshow(torchvision.utils.make_grid(concatenated))
print(example_batch[2].numpy())
