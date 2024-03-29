from data import augmentations
from data import dataset
from data import snapshot
from model import model,score

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
from torch.utils.data import Dataset, DataLoader, random_split,Subset
import torchvision
from torch import nn
from PIL import Image, ImageFilter
from torch.autograd import Variable
from torch.nn import Linear, ReLU, CrossEntropyLoss, Sequential, Conv2d, MaxPool2d, Module, Softmax, BatchNorm2d, Dropout
from torch.optim import Adam, SGD
from matplotlib import pyplot as plt
from skimage import io, transform
import cv2
import zipfile

import torch
from math import sin
import torchvision

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
from torch.utils.tensorboard import SummaryWriter
writer = SummaryWriter("runs/MRI")

random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
torch.cuda.manual_seed(0)
torch.backends.cudnn.deterministic = True

#https://www.youtube.com/watch?v=lIi69s8AaXA&feature=youtu.be

data = pd.read_csv('Set/kaggle_3m/data.csv')
data.head()
csv_path =   'Set/kaggle_3m/data.csv'
data_folder = 'Set/lgg-mri-segmentation/kaggle_3m'
eg_path = 'Set/lgg-mri-segmentation/kaggle_3m/TCGA_HT_8113_19930809'
eg_img = 'Set/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4944_20010208/TCGA_CS_4944_20010208_10.tif'
eg_mask = 'Set/lgg-mri-segmentation/kaggle_3m/TCGA_CS_4944_20010208/TCGA_CS_4944_20010208_10_mask.tif'

data = dataset.MySet(data_folder)
trainset, valset = random_split(data, [3600, 329])
train_loader = torch.utils.data.DataLoader(dataset=trainset, batch_size=10,shuffle=True)
val_loader = torch.utils.data.DataLoader(dataset=valset, batch_size=10)
model = model.Unet256((3,256,256)).to(device)
criterion = score.DiceBCELoss()
learning_rate = 1e-3
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
epochs = 25

train_loss = []
val_loss = []
PATH = '/model_data'
iteration_val = 0
iteration_train = 0
for epoch in range(epochs):
    print('Epoch {}/{}'.format(epoch + 1, epochs))
    start_time = time.time() 

    
    running_train_loss = []
    
    for image,mask in train_loader:
            optimizer.zero_grad() # setting gradient to zero
            iteration_train += 1
            image = image.to(device,dtype=torch.float)
            mask = mask.to(device,dtype=torch.float)
            pred_mask = model.forward(image) # forward propogation
            loss = criterion(pred_mask,mask)
            loss.backward()
            optimizer.step()
            running_train_loss.append(loss.item())
           
            writer.add_scalar('Loss/train',loss.item(),iteration_train)
            if iteration_train % 100==0:
                writer.add_image('image/train',image[0],iteration_train)
                writer.add_image('mask/train',mask[0],iteration_train)
                writer.add_image('pred_mask/train',pred_mask[0],iteration_train)

    else:
        running_val_loss = []
        
        with torch.no_grad():
            for image,mask in val_loader:
                iteration_val+=1
                image = image.to(device,dtype=torch.float)
                mask = mask.to(device,dtype=torch.float)                            
                pred_mask = model.forward(image)
                loss = criterion(pred_mask,mask)
                running_val_loss.append(loss.item())
                
                writer.add_scalar('Loss/val',loss.item(),iteration_val)
                if iteration_val%100==0:
                    writer.add_image('image/val',image[0],iteration_val)
                    writer.add_image('mask/val',mask[0],iteration_val)
                    writer.add_image('pred_mask/val',pred_mask[0],iteration_val)

    torch.save(model.state_dict(),PATH)
    epoch_train_loss = np.mean(running_train_loss) 
    print('Train loss: {}'.format(epoch_train_loss))                       
    train_loss.append(epoch_train_loss)
    
    epoch_val_loss = np.mean(running_val_loss)
    print('Validation loss: {}'.format(epoch_val_loss))                                
    val_loss.append(epoch_val_loss)
                      
    time_elapsed = time.time() - start_time
    print('{:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))