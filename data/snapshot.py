from torchvision import transforms as F
from torch.utils.data import Dataset, DataLoader, random_split
from PIL import Image
import torch

class Snapshot(Dataset):
    def __init__(self,image: Image, mask: Image):  
        self.image = image
        self.mask = mask
        
    @classmethod
    def pil_4_to_snapshot(cls, image_4: Image):
        image_4 = F.ToTensor()(image_4)
        image, mask = image_4[:3,:,:],\
                      image_4[-1,:,:]
        image, mask = F.ToPILImage()(image),\
                      F.ToPILImage()(mask)
        return cls(image,mask)
    
    @classmethod
    def tensor_4_to_snapshot(cls, tensor_4: torch.Tensor):
        img_tensor = tensor_4[:3, :, :]
        msk_tensor = tensor_4[-1, :, :]
        img_pil = F.ToPILImage()(img_tensor)
        msk_pil = F.ToPILImage()(msk_tensor)
        return cls(img_pil, msk_pil)
        
    def get_img_pil(self) ->Image:
        return self.image
    
    def get_mask_pil(self) ->Image:
        return self.mask
    
    
    def get_img_tensor(self) -> torch.Tensor:
        image = self.image
        image = image.resize((256,256))
        image = F.ToTensor()(image)
        return image
    
    def get_mask_tensor(self) -> torch.Tensor:
        mask = self.mask
        mask = mask.resize((256,256))
        mask =  F.ToTensor()(mask)
        return mask
    
    def to_pil_4(self) -> Image:
        image,mask = self.image,self.mask 
        image = F.ToTensor()(image)
        mask = F.ToTensor()(mask)
        snapshot = torch.cat((image,mask),0)
        snapshot = F.ToPILImage()(snapshot)
        return snapshot
    
    def to_tensor_4(self) -> torch.Tensor:
        image,mask = self.image,self.mask 
        image = F.ToTensor()(image)
        mask = F.ToTensor()(mask)
        tensor4 = torch.cat((image,mask),0)
        return tensor4