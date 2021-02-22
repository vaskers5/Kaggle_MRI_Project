import PIL
from PIL import Image
from torchvision.transforms import functional  as F
import random

class Augmentations:
    
    @staticmethod    
    def rotate_image(image):
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = F.rotate(image, angle)
        return image
    
    @staticmethod
    def h_flip(image):
        image = F.hflip(image)
        return image
    
    @staticmethod 
    def v_flip(image):
        image = F.vflip(image)
        return image
    
    @staticmethod  
    def add_contrast(image):
        image = F.adjust_contrast(image,2)
        return image
    
    
    @staticmethod
    def add_gamma(image):
        image = F.adjust_gamma(image,0.3,2)
        return image
    
    @staticmethod
    def add_blur(image):
        image = F.gaussian_blur(image,[1,3,2],[0.9,0.8])
        return image
    
    @staticmethod
    def transform(image):        
        transformations = [
                      Augmentations.rotate_image,
                      Augmentations.h_flip,
                       Augmentations.v_flip
#                       Augmentations.add_contrast,
#                       Augmentations.add_blur,
#                       Augmentations.add_gamma
            ]
        transformed_image = random.choice(transformations)(image)
        return transformed_image