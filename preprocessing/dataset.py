import os
import numpy as np
from tqdm import tqdm

import torch
from torch.utils.data import Dataset, DataLoader

import rasterio as rio
import albumentations as A

def get_aug_pipelines(means, stds): 
    # If a transform is provided, Albumentations applies it randomly with some prob,
    # The transformed version is returned to the model using __get_item__ method
    # The next time the same idx is sampled again (next epoch), the augmentation will likely be different
    # A.Compose([list of transforms]) Applies the contained transforms sequentially to the input data.

    # Albumentation requires (H, W, C) as input
    aug_pip = {
            'train': A.Compose([
                A.ShiftScaleRotate(shift_limit = 0.1, scale_limit=0.2, rotate_limit=30, p=0.2),
                A.HorizontalFlip(p=0.5),
                A.Normalize(mean=means, std=stds, max_pixel_value=1), # Standard scaling, other options: "min_max_per_channel" "min_max"
                A.ToTensorV2(transpose_mask=True),                  # And HWC -> CHW, #TODO BUT DOESNT WORK FOR MASK (remains H,W,C)
                ]),
            'val': A.Compose([
                A.Normalize(mean=means, std=stds, max_pixel_value=1), # img = (img - mean * max_pixel_value) / (std * max_pixel_value) so this formula first uses minmax scaling using max = 255 (default) and min = 0 so its in [0,1] range then applies standard scaling after using a mean and std in [0,1]
                A.ToTensorV2(transpose_mask=True),
                ]),
            'test': A.Compose([
                A.Normalize(mean=means, std=stds, max_pixel_value=1),
                A.ToTensorV2(transpose_mask=True),
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
    for f in tqdm(images, desc="Calculating mean and std for images"):
        f = os.path.join(image_dir, f)
        with rio.open(f) as src:
            img = torch.tensor(src.read(), dtype=torch.float) # Need to specify dtype otherwise its converted to torch.uint16
            means += torch.mean(img, dim=(1,2))
            stds += torch.std(img, dim=(1,2))

    means /= nimages
    stds /= nimages
    print("means", means)
    print("stds", stds)
    return means, stds
    
        
     
def one_hot_enc(mask, nclasses):
    encoded_mask = np.zeros((nclasses, mask.shape[1], mask.shape[2]), dtype=mask.dtype)
    
    for i in range(nclasses):
        temp_mask = mask[0]==i
        encoded_mask[i] = temp_mask.astype(mask.dtype) #TODO dont need .astype(mask.dtype) ?
    return encoded_mask

class RasterDataset(Dataset):
    def __init__(self, data_dir, RGBclasses, transform=None):
        self.root_dir = data_dir
        self.image_dir = os.path.join(data_dir, "images") 
        self.mask_dir = os.path.join(data_dir, "labels")
        self.images = [f for f in os.listdir(self.image_dir) if f.endswith(".tif")] 
        self.transform = transform
        self.RGBclasses = RGBclasses
        self.nclasses = len(self.RGBclasses)

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
        mask = one_hot_enc(mask, self.nclasses)
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

def create_dataloaders(data_dir, aug_pipelines, batch_size, num_workers, data_partition_list, RGBclasses): # data_dar = path to train val test sets
    datasets = {x: RasterDataset(data_dir=os.path.join(data_dir, x),      # Create datasets for train, val, test sets 
                                 RGBclasses=RGBclasses,
                                 transform=aug_pipelines[x]) for x in data_partition_list}
    dataloaders = {x: DataLoader(datasets[x],                             # Creates iterable for each set 
                                 batch_size = batch_size,
                                 num_workers = num_workers,
                                 shuffle=True,                               # After we iterate over all batches the data is shuffled
                                 drop_last = True) for x in data_partition_list} # If nsamples is not divisible by batch size, it drops remainder
    dataset_sizes = {x: len(datasets[x]) for x in data_partition_list}
    return dataloaders, dataset_sizes

if __name__ == "__main__":
    root_dir = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/train_val_set"
    patches_dir = os.path.join(root_dir, "train")
    
    dataset = RasterDataset(patches_dir)

    im, msk = dataset.__getitem__(10)

    print(im.dtype, msk.dtype, "msk shape", np.shape(msk), "img shape", np.shape(im))
    print("unique values of mask", np.unique(msk))
    #print(dataset.__dict__)
    
    # One-hot-enc func test
    arr = np.random.randint(0, 5, size=(1, 512, 512))
    arr = one_hot_enc(arr, len(dataset.RGBclasses))
    print(np.unique(arr[2]))
    print(np.shape(arr))

