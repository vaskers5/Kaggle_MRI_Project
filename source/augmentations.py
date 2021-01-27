from torchvision.transforms import functional  as F
import random

class Augmentations:
    def __init__(self,sample):
        self.image,self.mask = sample 
        
        

    def rotate_image(self):
        image, segmentation = self.image, self.mask
        if random.random() > 0.5:
            angle = random.randint(-30, 30)
            image = F.rotate(image, angle)
            segmentation = F.rotate(segmentation, angle)
        return image, segmentation
    
    def h_flip(self):
        image, segmentation = self.image, self.mask
        image = F.hflip(torch.Tensor)
        segmentation = F.hflip(torch.Tensor)
        return image,segmentation
    
    def v_flip(self):
        image, segmentation = self.image, self.mask
        image = F.vflip(image)
        segmentation = F.vflip(segmentation)
        return image,segmentation
        
    def add_contrast(self):
        image, segmentation = self.image, self.mask
        image = F.adjust_contrast(image,2)
        segmentation = F.adjust_contrast(segmentation,2)
        return image,segmentation
    
    def add_gamma(self):
        image, segmentation = self.image, self.mask
        image = F.adjust_gamma(image,0.3,2)
        segmentation = F.adjust_gamma(segmentation,0.3,2)
        return image,segmentation
    
    def add_blur(self):
        image, segmentation = self.image, self.mask
        image = F.gaussian_blur(image,[1,3,2],[0.9,0.8])
        segmentation = F.gaussian_blur(segmentation,[1,3,2],[0.9,0.8])
        
    
    def transform(self):        
        trasformations = {'0':rotate_image,
                      '2':h_flip,
                      '3':v_flip,
                      '4':add_contrast,
                      '5':add_blur,
                      '6':add_gamma,
                        }
        key = random.choice(list(transformations))
        transformed_image = transofmations[key](self)
        return transormed_image
        
        