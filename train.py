import torch
import torch.optim as optim
from torch.optim import lr_scheduler
import numpy as np

from tqdm import tqdm
import time

from models.UNet import UNET
from utils.helpers import display, debug_var, save_model, plot_training_results
from utils import metrics 
from preprocessing import dataset


DATA_PATH = "/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR/train_val_set"
SAVE_PATH = "/mnt/c/Users/Imesh/Desktop/summer_proj/models"

NUM_WORKERS = 6
NUM_EPOCHS = 30
BATCH_SIZE = 4
LEARNING_RATE = 0.001

MEANS = (417.9286, 405.0440, 415.4319)
STDS = (102.1648,  75.8841,  60.6446)


torch.manual_seed(42)
torch.cuda.manual_seed(42)

device = "cuda" if torch.cuda.is_available() else "cpu"

# Load dataloaders and data augmentation pipelines
aug_pipelines = dataset.get_aug_pipelines(means=MEANS, stds=STDS)
dataloaders, dataset_sizes = dataset.create_dataloaders(data_dir = DATA_PATH,
                                                        aug_pipelines = aug_pipelines,
                                                        batch_size = BATCH_SIZE,
                                                        num_workers = NUM_WORKERS,
                                                        ['train', 'val'],
                                                        )

# Create model
model = UNET(C_in = 3,
             C_out=3, 
             padding = 1)

model = model.to(device)

# Define metrics/loss func and define optimiser
metric_UNet = metrics.MeanIoU()
criterion_UNet = metrics.MultiDiceLoss()

optimizer = optim.Adam(model.parameters(), lr = LEARNING_RATE)

exp_lr_scheduler_UNet = lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.1)

# Print model's state_dict
print("Model's state_dict:")
for param_tensor in model.state_dict():
    print(param_tensor, "\t", model.state_dict()[param_tensor].size())

# Print optimizer's state_dict
print("Optimizer's state_dict:")
for var_name in optimizer.state_dict():
    print(var_name, "\t", optimizer.state_dict()[var_name])

"""
Train the model
"""
results = {
    "train_loss": [],
    "train_iou": [],
    "val_loss": [],
    "val_iou": []
}

for epoch in tqdm(range(NUM_EPOCHS), desc="Training Progress"):
    print(f"\nEpoch [{epoch+1}]/{NUM_EPOCHS}\n")

    for phase in ['train','val']:
        if phase == 'train':
            model.train()
        else:
            model.eval()

        running_loss = []
        running_iou = []

        # process batches
        for x, y in dataloaders[phase]:
            inputs = x.to(device)
            targets = y.to(device)
            
            if phase == 'train':
                optimizer.zero_grad()

            with torch.set_grad_enabled(phase == 'train'):
                outputs = model(inputs)

                loss = criterion_UNet(outputs, targets)
                iou = metric_UNet(outputs, targets)

                if phase == 'train':
                    loss.backward()
                    optimizer.step()

            running_loss.append(loss.item())
            running_iou.append(iou.item())
       
       # record results
        if phase == 'train':
            results['train_loss'].append(np.mean(running_loss))
            results['train_iou'].append(np.mean(running_iou))
        else:
            results['val_loss'].append(np.mean(running_loss))
            results['val_iou'].append(np.mean(running_iou))
    
    print(f'Train loss: {results["train_loss"][-1]} Train iou: {results["train_iou"][-1]} | Val loss: {results["val_loss"][-1]} Val iou: {results["val_iou"][-1]}')
    exp_lr_scheduler_UNet.step()


# TODO save model
save_model(model, SAVE_PATH, NUM_EPOCHS, LEARNING_RATE)

# TODO plot training results
plot_training_results(NUM_EPOCHS, results)
# TODO predict and evaluate













