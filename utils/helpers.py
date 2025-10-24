"""
Contains various utility functions used for model saving and data visualisation
"""
import os
import torch
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import matplotlib.pyplot as plt
from preprocessing.dataset import RasterDataset
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib.patches import Patch

plt.style.use('seaborn-v0_8-bright')
plt.rcParams["font.family"] = "serif"
plt.rcParams["axes.grid"] = False

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
    
    if torch.is_tensor(image):
        image = image.cpu().numpy()
    if torch.is_tensor(mask):
        mask = mask.cpu().numpy()
    
    #Normalize
    # Note: imshow expects [0, 1] floats or [0, 255] uint8
    image = image.astype(np.float32)
    for i in range(len(classes)):
        image[i] = (image[i] - image[i].min()) / (image[i].max() - image[i].min()) #TODO use min max values from whole image not for each patch 
    
    #print(np.shape(mask))    
    image = np.transpose(image, (1, 2, 0))
    # Convert one-hot encoded mask to RGB mask
    if mask.shape[0] == len(classes):# One-hot encoded (No need to check if shape == 3 since original mask is shape (1,H,W) anyway
        mask = np.argmax(mask, axis=0) # returns (H, W) find index of max value (1) along channel dim
    
    #print(np.shape(mask))    
    # Assign rgb values to mask_rgb
    mask_rgb = np.zeros((mask.shape[0], mask.shape[1], 3), dtype=np.uint8) # unint8 since we're using rgb values (0 255)
    for i, name in enumerate(classes):
        color = rgb_classes[name]
        mask_rgb[mask == i] = color
    
    # Normalize RGB mask to [0, 1] for display
    #mask_rgb_normalized = mask_rgb.astype(np.float32) / 255.0
    
    # Create the plot
    fig, axes = plt.subplots(1, 3, figsize=figsize)
    
    # Plot original image
    axes[0].imshow(image)
    axes[0].set_title('Image')
    #axes[0].axis('off')
    
    # Plot mask with custom colors
    axes[1].imshow(mask_rgb)
    axes[1].set_title('Ground truth')
    #axes[1].axis('off')
    
    # Create legend for mask
    legend_elements = [Patch(facecolor=np.array(color)/255.0, 
                           label=class_name) 
                     for class_name, color in rgb_classes.items()]
    axes[1].legend(handles=legend_elements, loc='upper right', 
                  bbox_to_anchor=(1, 1), fontsize=9)
    
    # Plot overlay
    axes[2].imshow(image)
    # Create a semi-transparent overlay with custom colors
    axes[2].imshow(mask_rgb, alpha=0.5)
    axes[2].set_title('Image & Mask Overlay')
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

def plot_predictions(predictions, rgb_classes, pdf_path="/mnt/c/Users/Imesh/Desktop/summer_proj/code/images/preds.pdf"):
    """
    Function for plotting val and test predictions
    """
    with PdfPages(pdf_path) as pdf:
        for batch_idx, (images, true_masks, pred_masks) in enumerate(predictions):
            for i in range(images.shape[0]):
                
                fig, axes = plt.subplots(1, 3, figsize=(15, 5))
                # Denormalize image to [0,1]
                img = images[i].permute(1, 2, 0).numpy()
                img = (img - img.min()) / (img.max() - img.min())
                
                # Convert one-hot true mask to class indices
                true_mask_np = torch.argmax(true_masks[i], dim=0).numpy()  #(C,H,W) -> (H,W)
                pred_mask_np = pred_masks[i].numpy()  # (H,W)
                
                # Convert masks to RGB
                true_mask_rgb = np.zeros((true_mask_np.shape[0], true_mask_np.shape[1], 3), dtype=np.uint8)
                pred_mask_rgb = np.zeros((pred_mask_np.shape[0], pred_mask_np.shape[1], 3), dtype=np.uint8)
                
                for class_idx, color in enumerate(rgb_classes.values()):
                    true_mask_rgb[true_mask_np == class_idx] = color
                    pred_mask_rgb[pred_mask_np == class_idx] = color
                
                # Create legend
                legend_elements = [Patch(facecolor=np.array(color)/255.0, label=class_name)
                                 for class_name, color in rgb_classes.items()]
                
                # Plot
                axes[0].imshow(img)
                axes[0].set_title('Image')
                #axes[0].axis('off')
                
                axes[1].imshow(img)
                axes[1].imshow(true_mask_rgb, alpha=0.8)
                axes[1].set_title('Ground Truth')
                #axes[1].axis('off')
                axes[1].legend(handles=legend_elements, loc='upper right', 
                              bbox_to_anchor=(1, 1), fontsize=9)
                
                axes[2].imshow(img)
                axes[2].imshow(pred_mask_rgb, alpha=0.8)
                axes[2].set_title('Prediction')
                #axes[2].axis('off')
                axes[2].legend(handles=legend_elements, loc='upper right', 
                              bbox_to_anchor=(1, 1), fontsize=9)
                
                plt.tight_layout()
                pdf.savefig(fig, bbox_inches='tight')
                plt.close(fig)

    print(f"Saved predictions to {pdf_path}")             

def save_model(model, save_dir, epoch, lr):
    checkp_name = f"UNet_model_NIR_epoch_{epoch}_lr_{lr}.pth"
    save_path = os.path.join(save_dir, checkp_name)
    print("-"*20)
    print("Saving model as ", checkp_name)
    print("-"*20)
    torch.save(model.state_dict(), save_path)

    
def plot_training_results(epochs,
                          results,
                          save_dir="/mnt/c/Users/Imesh/Desktop/summer_proj/code/images",
                          filename="training_results.png"):
    
    # Create epoch list
    epoch_range = list(range(1, epochs + 1))
    
    # Plot everything
    plt.figure(figsize=(10, 6))
    plt.plot(epoch_range, results["train_loss"], label="Train Loss", linewidth=1.5)
    plt.plot(epoch_range, results["val_loss"], label="Validation Loss", linewidth=1.5)
    plt.plot(epoch_range, results["train_iou"], label="Train IoU", linewidth=1.5)
    plt.plot(epoch_range, results["val_iou"], label="Validation IoU", linewidth=1.5)
    
    plt.xlabel("Epoch")
    plt.ylabel("Value")
    plt.title("Training and Validation Metrics")
    plt.legend()
    plt.grid(True)
    
    # Save the figure
    save_path = os.path.join(save_dir, filename)
    plt.savefig(save_path, bbox_inches='tight')
    plt.close()
    print(f"Plot saved to {save_path}")

    
# Function for debugging
def debug_var(x, name="Variable"):
    
    print(f"\n Debugging: {name}")
    print("-" * 40)

    # Type and basic info
    print(f"Type: {type(x)}")

    # Handle PyTorch tensor
    if isinstance(x, torch.Tensor):
        print(f"Shape: {tuple(x.shape)}")
        print(f"Dtype: {x.dtype}")
        print(f"Device: {x.device}")
        print(f"Requires Grad: {x.requires_grad}")
        print(f"Min: {x.amin(dim=(1,2))}")
        print(f"Max: {x.amax(dim=(1,2))}")
        print(f"Mean: {torch.mean(x, dim = (1,2))}")
        print(f"Std: {torch.std(x, dim = (1,2))}")
        print(f"Has NaNs: {torch.isnan(x).any().item()}")

    # Handle NumPy array
    elif isinstance(x, np.ndarray):
        print(f"Shape: {x.shape}")
        print(f"Dtype: {x.dtype}")
        print(f"Min: {np.nanmin(x)}")
        print(f"Max: {np.nanmax(x)}")
        print(f"Has NaNs: {np.isnan(x).any()}")

    else:
        print("⚠️ Unsupported type")

    print("-" * 40)


if __name__ == "__main__":
    root_dir = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/train_val_set"
    patches_dir = os.path.join(root_dir, "train")
    dataset = RasterDataset(patches_dir, training = None)

    num_patches = dataset.__len__()
    idx = np.random.randint(0,num_patches)
    print(idx)
    print(dataset.images[idx])

    image, mask = dataset.__getitem__(idx) 
    print(f"image: ({image[0].max()}, {image[0].min()}), ({image[1].max()}, {image[1].min()}), ({image[2].max()}, {image[2].min()})")
    display(image, mask, dataset.RGBclasses, dataset.classes, figsize=(15, 5))
