"""
Contains various utility functions used for model saving and data visualisation
"""
import os
import torch
import numpy as np
#import torch.nn.functional as F
#import torch.nn as nn
import matplotlib.pyplot as plt
from preprocessing.dataset import RasterDataset

# Save the model to the target dir
#def save_model(model: torch.nn.Module, target_dir: str, epoch: int):
#def plot_curve(results: dict, epochs: int):

# Helper func for img vis
def display(image, mask, rgb_classes, classes, figsize=(15, 5)):
    """
    Plot pair of image and mask for visualisation
    with rgb colours for mask
    
    Args:
        image (numpy.ndarray): Image (C, H, W) (could be tensor as well)
        mask (numpy.ndarray): One-hot encoded mask tensor (C, H, W)
        rgb_classes (dict): Dictionary with class names and RGB colors
        classes (list): List of class names in order
        figsize (tuple): Fig size tuple
    """
    

    image = np.transpose(image, (1, 2, 0))
    
    #Normalize
    # Note: imshow expects [0, 1] floats or [0, 255] uint8
    image = image.astype(np.float32)
    for i in range(len(classes)):
        image[i] = (image[i] - image[i].min()) / (image[i].max() - image[i].min()) 
    
    # Convert one-hot encoded mask to RGB mask
    mask_rgb = np.zeros((mask.shape[1], mask.shape[2], 3), dtype=np.uint8)
    if mask.shape[0] == len(classes):# One-hot encoded (No need to check if shape == 3 since original mask is shape (1,H,W) anyway
        mask = np.argmax(mask, axis=0) # returns (H, W) find index of max value (1) along channel dim
    
        
    # Assign rgb values to mask_rgb
    for i, name in enumerate(classes):
        color = rgb_classes[name]
        mask_rgb[mask == i] = color
    
    # Normalize RGB mask to [0, 1] for display
    #mask_rgb_normalized = mask_rgb.astype(np.float32) / 255.0
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Input Image')
    #axes[0].axis('off')
    
    # Plot mask with custom colors
    axes[1].imshow(mask_rgb)
    axes[1].set_title('Segmentation Mask')
    #axes[1].axis('off')
    
    # Create legend for mask
    from matplotlib.patches import Patch
    legend_elements = [Patch(facecolor=np.array(color)/255.0, 
                           label=class_name) 
                     for class_name, color in rgb_classes.items()]
    axes[1].legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(1, 1), fontsize=8)
    
    # Plot overlay
    axes[2].imshow(image)
    # Create a semi-transparent overlay with custom colors
    axes[2].imshow(mask_rgb, alpha=0.5)
    axes[2].set_title('Image + Mask Overlay')
    #axes[2].axis('off')
    
    plt.tight_layout()
    plt.show()

# Simplified function that works directly with dataset instance
def plot_dataset_sample(dataset, index=0):
    """
    Simple function to plot a single sample from the dataset.
    
    Args:
        dataset: RasterDataset instance
        index (int): Index of sample to plot
    """
    image, mask = dataset[index]
    display(image, mask, rgb_classes=dataset.RGBclasses, 
                        classes=dataset.classes)

    
    
# Categorical Cross Entropy Loss
#class CategoricalCrossEntropyLoss(nn.Module):
#    def __init__(self):
#        super().__init__()

# Multiclass Dice Loss
#class MultiDiceLoss(nn.Module):
#    def __init__(self):
#        super().__init__()
    

# Mean IoU Score
#class MeanIoU(nn.Module):
#    def __init__(self):
#        super().__init__()
    

if __name__ == "__main__":
    root_dir = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR"
    patches_dir = os.path.join(root_dir, "patches")
    dataset = RasterDataset(patches_dir, training = None)

    num_patches = dataset.__len__()
    idx = np.random.randint(0,num_patches)
    print(idx)
    print(dataset.images[idx])

    image, mask = dataset.__getitem__(idx) 
    print(f"image: ({image[0].max()}, {image[0].min()}), ({image[1].max()}, {image[1].min()}), ({image[2].max()}, {image[2].min()})")
    display(image, mask, dataset.RGBclasses, dataset.classes, figsize=(15, 5))
