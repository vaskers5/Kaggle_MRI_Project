import torch
import torchvision
from torchvision import transforms
from torchvision.transforms import ToPILImage
from torchvision.transforms import ToTensor
from torch.utils.data import Dataset
import PIL 
from PIL import Image

class Snapshot(Dataset):
    def __init__(self,image: Image, mask: Image):  
        self.image = image
        self.mask = mask
        
    def get_img_pil(self) ->Image:
        return self.image
    
    def get_mask_pil(self) ->Image:
        return self.mask
    
    def get_snapshot_pil(self,snapshot: torch.Tensor) -> Image:
        snapshot = ToPILImage()(snapshot)
        return snapshot
    
    def get_img_tensor(self) -> torch.Tensor:
        image = self.image
        image = transform.resize(image,(256,256))
        image = image / 255
        image = image.transpose((2, 0, 1))
        image = torch.from_numpy(image)
        return image
    
    def get_mask_tensor(self) -> torch.Tensor:
        image = self.mask
        mask = transform.resize(mask,(256,256))
        mask = mask / 255
        mask = np.expand_dims(mask,axis=-1).transpose((2, 0, 1))
        mask = torch.from_numpy(mask)
        return mask
    
    def to_tensor_4(self) -> torch.Tensor:
        self.image,self.mask = image,mask
        image = ToTensor()(image)
        mask = ToTensor()(mask)
        snapshot = torch.cat((image,mask),0)
        return snapshot