import os
import torch
import numpy as np
from preprocessing.dataset import RasterDataset, mean_std, get_aug_pipeline
from utils.helpers import display, debug_var
import albumentations as A


if __name__ == "__main__":
    root_dir = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/train_val_set"
    patches_dir = os.path.join(root_dir, "train")
    images_dir = os.path.join(patches_dir, "images")
    # calc means and std
    #means, stds = mean_std(images_dir) #means [417.9286, 405.0440, 415.4319] stds [102.1648,  75.8841,  60.6446]
    means = (417.9286, 405.0440, 415.4319)
    stds = (102.1648,  75.8841,  60.6446)
    #means = (0, 0, 0)
    #stds = (1,  1, 1)
    # augmentation dict
    aug_pipelines = get_aug_pipelines(means, stds)
    dataset = RasterDataset(patches_dir, transform = aug_pipelines['train'])
    dataset2 = RasterDataset(patches_dir, transform = None)
    
    num_patches = len(dataset) # runs __len()__ method
    idx = np.random.randint(0,num_patches)
    print(idx)
    print(dataset.images[idx])

    image, mask = dataset[idx]
    image2, mask2 = dataset2[idx] # runs __getitem__() method
    #mask = mask.permute(2,0,1)
    debug_var(image, "image")
    debug_var(image2, "image2")
    debug_var(mask, "mask")
    debug_var(mask2, "mask2")
    display(image, mask, dataset.RGBclasses, dataset.classes, figsize=(15, 5))
    display(image2, mask2, dataset.RGBclasses, dataset.classes, figsize=(15, 5))




