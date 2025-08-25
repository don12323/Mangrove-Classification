import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import rasterio
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt

class DoubleConv(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class MangroveUNet(nn.Module):
    def __init__(self, in_channels=7):  # 4 bands + NDVI + NDMI + NDWI
        super().__init__()
        
        # Encoder
        self.conv1 = DoubleConv(in_channels, 64)
        self.conv2 = DoubleConv(64, 128)
        self.conv3 = DoubleConv(128, 256)
        self.conv4 = DoubleConv(256, 512)
        
        # Bridge
        self.bridge = DoubleConv(512, 1024)
        
        # Decoder
        self.up_conv4 = nn.ConvTranspose2d(1024, 512, kernel_size=2, stride=2)
        self.dconv4 = DoubleConv(1024, 512)
        
        self.up_conv3 = nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2)
        self.dconv3 = DoubleConv(512, 256)
        
        self.up_conv2 = nn.ConvTranspose2d(256, 128, kernel_size=2, stride=2)
        self.dconv2 = DoubleConv(256, 128)
        
        self.up_conv1 = nn.ConvTranspose2d(128, 64, kernel_size=2, stride=2)
        self.dconv1 = DoubleConv(128, 64)
        
        self.output = nn.Conv2d(64, 1, kernel_size=1)
        self.maxpool = nn.MaxPool2d(2)
        
    def forward(self, x):
        # Encoder
        conv1 = self.conv1(x)
        pool1 = self.maxpool(conv1)
        
        conv2 = self.conv2(pool1)
        pool2 = self.maxpool(conv2)
        
        conv3 = self.conv3(pool2)
        pool3 = self.maxpool(conv3)
        
        conv4 = self.conv4(pool3)
        pool4 = self.maxpool(conv4)
        
        # Bridge
        bridge = self.bridge(pool4)
        
        # Decoder
        up4 = self.up_conv4(bridge)
        up4 = torch.cat([up4, conv4], dim=1)
        up4 = self.dconv4(up4)
        
        up3 = self.up_conv3(up4)
        up3 = torch.cat([up3, conv3], dim=1)
        up3 = self.dconv3(up3)
        
        up2 = self.up_conv2(up3)
        up2 = torch.cat([up2, conv2], dim=1)
        up2 = self.dconv2(up2)
        
        up1 = self.up_conv1(up2)
        up1 = torch.cat([up1, conv1], dim=1)
        up1 = self.dconv1(up1)
        
        return torch.sigmoid(self.output(up1))

class MangroveDataset(Dataset):
    def __init__(self, patches, masks):
        self.patches = torch.FloatTensor(patches)
        self.masks = torch.FloatTensor(masks)
    
    def __len__(self):
        return len(self.patches)
    
    def __getitem__(self, idx):
        return self.patches[idx], self.masks[idx]

class DataProcessor:
    def __init__(self, image_path, mask_path, patch_size=256, stride=128):
        self.image_path = image_path
        self.mask_path = mask_path
        self.patch_size = patch_size
        self.stride = stride
        
    def calculate_indices(self, image):
        """Calculate spectral indices"""
        red = image[..., 0]
        green = image[..., 1]
        blue = image[..., 2]
        nir = image[..., 3]
        
        # Calculate NDVI
        ndvi = np.zeros_like(red)
        valid = (red != 0) & (nir != 0)
        ndvi[valid] = (nir[valid] - red[valid]) / (nir[valid] + red[valid] + 1e-8)
        
        # Calculate NDMI (Normalized Difference Mud Index)
        # Using green and red bands for mud detection
        ndmi = np.zeros_like(red)
        valid = (green != 0) & (red != 0)
        ndmi[valid] = (green[valid] - red[valid]) / (green[valid] + red[valid] + 1e-8)
        
        # Calculate NDWI (Normalized Difference Water Index)
        ndwi = np.zeros_like(red)
        valid = (green != 0) & (nir != 0)
        ndwi[valid] = (green[valid] - nir[valid]) / (green[valid] + nir[valid] + 1e-8)
        
        return ndvi, ndmi, ndwi
    
    def load_data(self):
        """Load NEO image and mask"""
        with rasterio.open(self.image_path) as src:
            # Read first 4 bands
            image = np.stack([src.read(i) for i in range(1, 5)], axis=-1)
            
            # Calculate indices
            ndvi, ndmi, ndwi = self.calculate_indices(image)
            
            # Stack all bands and indices
            image = np.dstack((image, ndvi, ndmi, ndwi))
        
        with rasterio.open(self.mask_path) as src:
            mask = src.read(1)
            mask = np.expand_dims(mask, axis=-1)
        
        return image, mask
    
    def extract_patches(self, image, mask):
        """Extract patches from image and mask"""
        patches = []
        mask_patches = []
        
        for i in range(0, image.shape[0] - self.patch_size + 1, self.stride):
            for j in range(0, image.shape[1] - self.patch_size + 1, self.stride):
                patch = image[i:i+self.patch_size, j:j+self.patch_size, :]
                mask_patch = mask[i:i+self.patch_size, j:j+self.patch_size, :]
                
                # Only keep patches with data
                if np.any(patch != 0):
                    patches.append(patch)
                    mask_patches.append(mask_patch)
        
        # Reshape for PyTorch (N, C, H, W)
        patches = np.array(patches).transpose(0, 3, 1, 2)
        mask_patches = np.array(mask_patches).transpose(0, 3, 1, 2)
        
        return patches, mask_patches
    
    def preprocess_data(self):
        """Load and preprocess all data"""
        # Load data
        image, mask = self.load_data()
        
        # Extract patches
        patches, mask_patches = self.extract_patches(image, mask)
        
        # Normalize image data
        patches = patches.astype(np.float32)
        for i in range(patches.shape[1]):  # Normalize each channel
            channel = patches[:, i, :, :]
            mean = np.mean(channel[channel != 0])
            std = np.std(channel[channel != 0])
            patches[:, i, :, :] = (channel - mean) / (std + 1e-8)
        
        # Split into train and validation sets
        X_train, X_val, y_train, y_val = train_test_split(
            patches, mask_patches, test_size=0.2, random_state=42
        )
        
        return X_train, X_val, y_train, y_val

def train_model(model_path='mangrove_neo_data.tif', mask_path='mangrove_labels.tif', 
                device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Train the U-Net model"""
    # Initialize data processor and model
    processor = DataProcessor(model_path, mask_path)
    X_train, X_val, y_train, y_val = processor.preprocess_data()
    
    # Create datasets and dataloaders
    train_dataset = MangroveDataset(X_train, y_train)
    val_dataset = MangroveDataset(X_val, y_val)
    
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=8)
    
    # Initialize model and move to device
    model = MangroveUNet().to(device)
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters())
    
    # Training loop
    num_epochs = 100
    best_val_loss = float('inf')
    train_losses = []
    val_losses = []
    
    for epoch in range(num_epochs):
        model.train()
        epoch_loss = 0
        
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            
            optimizer.zero_grad()
            output = model(data)
            loss = criterion(output, target)
            loss.backward()
            optimizer.step()
            
            epoch_loss += loss.item()
            
        train_losses.append(epoch_loss / len(train_loader))
        
        # Validation
        model.eval()
        val_loss = 0
        
        with torch.no_grad():
            for data, target in val_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                val_loss += criterion(output, target).item()
        
        val_loss /= len(val_loader)
        val_losses.append(val_loss)
        
        print(f'Epoch {epoch+1}, Train Loss: {train_losses[-1]:.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
    
    return model, train_losses, val_losses

def predict(model, image_path, device='cuda' if torch.cuda.is_available() else 'cpu'):
    """Predict on a new image"""
    processor = DataProcessor(image_path, None)
    image, _ = processor.load_data()
    
    # Process image in patches
    patches, _ = processor.extract_patches(image, np.zeros_like(image[..., :1]))
    
    # Create dataset and dataloader
    dataset = torch.FloatTensor(patches)
    loader = DataLoader(dataset, batch_size=8)
    
    model.eval()
    predictions = []
    
    with torch.no_grad():
        for batch in loader:
            batch = batch.to(device)
            output = model(batch)
            predictions.append(output.cpu().numpy())
    
    return np.concatenate(predictions)

if __name__ == "__main__":
    # Train model
    NEO_path = '/mnt/c/Users/Imesh/Desktop/summer_proj/MAPQ3389-EnSTAR'
    sixbands_path = os.path.join(NEO_path, '1-21-2022_Ortho_6Band.tif')
    label_path = os.path.join(NEO_path,'ndvi.tif') 
    model, train_losses, val_losses = train_model(
            model_path=sixbands_path,
            mask_path=label_path)
    
    # Plot training history
    plt.figure(figsize=(10, 5))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Model Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()
