import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import os
from statistics import mean
from torch.autograd import Variable
import time
from PIL import Image
import random
from skimage import io, transform
from collections import OrderedDict
import numpy as np
import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torch import nn
from PIL import Image, ImageFilter
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from matplotlib import pyplot as plt
from skimage import io, transform
from .Data_Set import My_set
from .Visual import image_convert
from .Visual import mask_convert
from .Visual import plot_img

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')


random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

data = pd.read_csv('Set/kaggle_3m/data.csv')
data.head()
csv_path =   'Set/kaggle_3m/data.csv'
data_folder = 'Set/lgg-mri-segmentation/kaggle_3m'
eg_path = 'Set/lgg-mri-segmentation/kaggle_3m/TCGA_HT_8113_19930809'
eg_img = 'Set/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4944_20010208/TCGA_CS_4944_20010208_10.tif'
eg_mask = 'Set/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4944_20010208/TCGA_CS_4944_20010208_10_mask.tif'

# чел мешает трейн и валидацию

trainset, valset = random_split(data, [3600, 329])

train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=10,shuffle=True)

val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=10)


