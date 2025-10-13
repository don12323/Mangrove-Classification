import os
import numpy as np

import torch
from torch.utils.data import Dataset, DataLoader

import rasterio as rio
import albumentations as A

def get_aug_pipeline(mean, std): 
    # If a transform is provided, Albumentations applies it randomly with some prob,
    # The transformed version is returned to the model using __get_item__ method
    # The next time the same idx is sampled again (next epoch), the augmentation will likely be different
    # A.Compose([list of transforms]) Applies the contained transforms sequentially to the input data.

    # Albumentation requires (H, W, C) as input
    aug_pip = {
            'train': A.Compose([
                A.ShiftScaleRotate(shift_limit = 0.1, scale_limit=0.2, rotate_limit=30, p=0.2),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=mean, std=std), # Standard scaling
                A.ToTensorV2(),                  # And HWC -> CHW
                ]),
            'val': A.Compose([
                A.Normalize(mean=mean, std=std), # img = (img - mean * max_pixel_value) / (std * max_pixel_value)
                A.ToTensorV2(),
                ]),
            'test': A.Compose([
                A.Normalize(mean=mean, std=std),
                A.ToTensorV2(),
            ]),
    }
    return aug_pip

def mean_std(image_dir):
    # Calculate mean and std for each channel for training dataset
    # Later used for standardising train val test images
    # Returns tensors 
    images = os.listdir(image_dir)
    print(images)
    means = torch.zeros(3) # TODO can albumentation take in torch.float?
    stds = torch.zeros(3)
    nimages = len(images)
    for f in images:
        with rio.open(f) as src:
            img = torch.tensor(src.read())
            means += torch.mean(img, dim=(1,2))
            stds += torch.std(img, dim=(1,2))

    means = means/nimages
    stds = stds/nimages
    
    return means, stds
    
        
     
def one_hot_enc(mask, classes):
    encoded_mask = np.zeros((len(classes), mask.shape[1], mask.shape[2]), dtype=mask.dtype)
    
    for i in range(len(classes)):
        temp_mask = mask[0]==i
        encoded_mask[i] = temp_mask.astype(mask.dtype)
    return encoded_mask

class RasterDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.root_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images") 
        self.mask_dir = os.path.join(data_dir, "labels")
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith(".tif")] 
        self.transform = transform

        self.RGBclasses = {
                'Nodata': [155,155,155],
                'Water': [58, 221, 254],
                'Mangrove': [66,242,30]
                }
        self.classes = ['Nodata', 'Water', 'Mangrove']

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.images[idx])
        mask_path = os.path.join(self.mask_dir, self.images[idx].replace("patch", "mask"))

        #Read data, normalize, one-hot encode, data augmentation
        with rio.open(img_path) as src:
            image = src.read()             # dtype is Uint16 (C, H, W)
        with rio.open(mask_path) as src:
            mask = src.read()              # dtype is Uint16 (C, H, W)
        
        # One hot encoding
        mask = one_hot_enc(mask, self.classes)
        # Convert to float as albumentation takes in [uint8 or float32] (using float32 is much slower compared to uint)
        image, mask = image.astype(np.float32), mask.astype(np.float32)
        
        # Transform
        if self.transform is not None:
            # transpose data to be H,W,C as albumentation expects
            image, mask = image.transpose(1,2,0), mask.transpose(1,2,0)
            transformed = self.transform(image=image, mask=mask)
            image, mask = transformed['image'], transformed['mask']
        else:
            #TODO if no transform just convert to tensor and maybe normalize
            # Cant use transforms.ToTensor since its only for PIL images
            image, mask = torch.tensor(image), torch.tensor(mask)
        
        return image, mask

if __name__ == "__main__":
    root_dir = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/train_val_set"
    patches_dir = os.path.join(root_dir, "train")
    
    dataset = RasterDataset(patches_dir, training = None)

    im, msk = dataset.__getitem__(10)

    print(im.dtype, msk.dtype, "msk shape", np.shape(msk), "img shape", np.shape(im))
    print("unique values of mask", np.unique(msk))
    #print(dataset.__dict__)
    
    # One-hot-enc func test
    arr = np.random.randint(0, 5, size=(1, 512, 512))
    arr = one_hot_enc(arr, dataset.classes)
    print(np.unique(arr[2]))
    print(np.shape(arr))

