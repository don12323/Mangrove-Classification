import torch
import numpy as np
import random 
from tqdm import tqdm
import os

from utils.helpers import plot_predictions
from utils import metrics
from models.UNet import UNET

from preprocessing import dataset

def main():
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
    MODEL_PATH = "/mnt/c/Users/Imesh/Desktop/summer_proj/models/UNet_model_NIR_epoch_30_lr_0.001.pth"
    DATA_PATH = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/train_val_set"

    MEANS = (417.9286, 405.0440, 415.4319)
    STDS = (102.1648,  75.8841,  60.6446)
    print("here")
    # Load dataloaders and data augmentation pipelines
    aug_pipelines = dataset.get_aug_pipelines(means=MEANS, stds=STDS)
    partitions = ['val', 'test']
    dataloaders, dataset_sizes = dataset.create_dataloaders(data_dir = DATA_PATH,
                                                            aug_pipelines = aug_pipelines,
                                                            batch_size = 6,
                                                            num_workers = 6,
                                                            data_partition_list=['val', 'test'],
                                                            )
    # Load model
    model = UNET(C_in=3, C_out=3, padding=1)
    checkp = torch.load(MODEL_PATH, weights_only=True)
    model.load_state_dict(checkp)
    model.to(DEVICE)
    
    # Predict
    batches = get_random_batches(dataloaders['test'])
    predictions = predict_mask(batches, model, device=DEVICE)

    #print(dir(dataloaders['test']))
    # Visualize and save
    plot_predictions(predictions, rgb_classes=dataloaders['test'].dataset.RGBclasses)
    
    # Evaluate on the test set
    mean_loss, mean_iou, class_ious = eval_model(model=model,
                                     dataloader=dataloaders['test'],
                                     metric=metrics.MeanIoU(),
                                     loss_fn=metrics.MultiDiceLoss(),
                                     device=DEVICE,
                                     )
    mean_class_ious = np.mean(class_ious, axis=0)
    std_class_ious = np.std(class_ious, axis=0)

    print(f"class ious: Nodata: {mean_class_ious[0]} \pm {std_class_ious[0]}, Water: {mean_class_ious[1]} \pm {std_class_ious[1]}, Mang: {mean_class_ious[2]} \pm {std_class_ious[2]}")
    print(f"Test set evaluation: Multi Dice loss: {mean_loss}, Mean IoU: {mean_iou}")

def eval_model(model, dataloader, metric, loss_fn, device):#TODO change metric into metrics and calculate a bunch
    """
    Evaluate the model performace on test set after training

    """
    model.eval()
    model.to(device)

    running_ious = []
    running_losses = []
    running_class_ious = []
    for x, y in dataloader:
        inputs = x.to(device)
        targets = y.to(device)

        with torch.no_grad():
            outputs = model(inputs)

            # Calc loss
            loss = loss_fn(outputs, targets)
            running_losses.append(loss.item())
            # Metric
            iou = metric(outputs, targets)
            class_ious = metric.get_class_iou(outputs, targets)
            running_class_ious.append(class_ious)
            print(class_ious)
            running_ious.append(iou.item())
    
    return np.mean(running_losses), np.mean(running_ious), np.array(running_class_ious)

def get_random_batches(dataloader, num_batches=8):
    """Get random batches from dataloader"""
    all_batches = list(dataloader)
    selected_batches = random.sample(all_batches, min(num_batches, len(all_batches)))
    return selected_batches

def predict_mask(batches, model, device):
    """
    makes predictions on a number of batches
    """

    # Check if model is in device, else set to device

    model.eval()
    predictions = []
    
    with torch.no_grad():
        for images, masks in tqdm(batches, desc="Predicting", unit="batch"):
            images = images.to(device)
            outputs = model(images)

            pred_masks = torch.argmax(outputs, dim = 1) # (N,C,H,W)->(N,H,W)
            predictions.append((images.cpu(), masks, pred_masks.cpu()))
    return predictions



if __name__ == "__main__":
    main()
