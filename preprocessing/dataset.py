import os
import numpy as np

import torch
from torch.utils.data import Dataset

import rasterio as rio

def one_hot_enc(mask, classes):
    encoded_mask = np.zeros((len(classes), mask.shape[1], mask.shape[2]), dtype=mask.dtype)
    
    for i in range(len(classes)):
        temp_mask = mask[0]==i
        encoded_mask[i] = temp_mask.astype(mask.dtype)
    return encoded_mask

class RasterDataset(Dataset):
    def __init__(self, data_dir, training, transform=None):
        self.root_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images") 
        self.mask_dir = os.path.join(data_dir, "labels")
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith(".tif")] 
        self.training = training
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
             image = src.read()
        with rio.open(mask_path) as src:
            mask = src.read()
        # One hot encoding
        mask = one_hot_enc(mask, self.classes)
        # Transform


        return image, mask

if __name__ == "__main__":
    root_dir = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR"
    patches_dir = os.path.join(root_dir, "patches")
    
    dataset = RasterDataset(patches_dir, training = None)

    im, msk = dataset.__getitem__(10)

    print(im.dtype, msk.dtype, "msk shape", np.shape(msk), "img shape", np.shape(im))
    print("unique values of mask", np.unique(msk))
    #print(dataset.__dict__)
    
    # One-hot-enc func test
    arr = np.random.randint(0, 5, size=(1, 512, 512))
    arr = one_hot_enc(arr)
    print(np.unique(arr[2]))
    print(np.shape(arr))
