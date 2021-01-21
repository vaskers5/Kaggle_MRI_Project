import numpy as np
from skimage import io, transform
import torchvision
import torch


class My_set(Dataset):
    def __init__(self,path):
        self.path = path
        self.patients = [file for file in os.listdir(path)
                         if file not in ['data.csv','README.md']] #список пациентов
        self.masks,self.images = [],[] #массивы масок и изображений для каждого элемента
        
        for patient in self.patients:
            for file in os.listdir(os.path.join(self.path,patient)): #по каждому пациенту находят маски и изображения
                if 'mask' in file.split('.')[0].split('_'):
                    self.masks.append(os.path.join(self.path,
                                                   patient,file))
                else: 
                    self.images.append(os.path.join(self.path,patient,file)) 
          
    def __len__(self):
        return len(self.images)
    
    def augmentation(self,image):
        transforms = torchvision.transforms.Compose([
    torchvision.transforms.Resize((224,224)),
    torchvision.transforms.ColorJitter(hue=.05, saturation=.05),
    torchvision.transforms.RandomHorizontalFlip(),
    ])
        return image

    
    def __getitem__(self,index,aug = False):
        image = self.images[index]
        mask = self.masks[index]
        
        image = io.imread(image)
        if aug:
            image  = self.augmentation(image)
            
""""вся срань ниже нужна для преобразования картинок
без этой хуйни ничего работать дальше не будет"""

        image = transform.resize(image,(256,256))
        image = image / 255
        image = image.transpose((2, 0, 1))
        
        image = torch.from_numpy(image)
#аналагично комментариям выше
        mask = io.imread(mask)
        mask = transform.resize(mask,(256,256))
        mask = mask / 255
        mask = np.expand_dims(mask,axis=-1).transpose((2, 0, 1))
        mask = torch.from_numpy(mask)
        return (image, mask)
    