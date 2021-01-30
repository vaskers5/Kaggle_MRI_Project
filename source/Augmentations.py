import PIL
from PIL import Image
from torchvision.transforms import functional  as F
import random

class Augmentations:
    def __init__(self,sample):
        self.new_image = sample 
        
    def rotate_image(self):
        image = self.new_image
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = F.rotate(image, angle)
        return image
    
    def h_flip(self):
        image = self.new_image
        image = F.hflip(image)
        return image
    
    def v_flip(self):
        image = self.new_image
        image = F.vflip(image)
        return image
        
    def add_contrast(self):
        image = self.new_image
        image = F.adjust_contrast(image,2)
        return image
    
    def add_gamma(self):
        image = self.new_image
        image = F.adjust_gamma(image,0.3,2)
        return image
    
    def add_blur(self):
        image = self.new_image
        image = F.gaussian_blur(image,[1,3,2],[0.9,0.8])
        return image
    
    def transform(self):        
        transformations = [
                      self.rotate_image,
                      self.h_flip,
                      self.v_flip,
                      self.add_contrast,
                      self.add_blur,
                      self.add_gamma
            ]
        transformed_image = random.choice(transformations)()
        return transformed_image
        
        