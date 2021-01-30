import PIL
from PIL import Image

class DataLoader:
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
    
    
    def __getitem__(self,index):
        image = self.images[index]
        mask = self.masks[index]
        
        image = Image.open(image)
        mask = Image.open(mask)
        return image,mask