import torch
import torch.nn as nn
import torchvision.transforms.functional as TF
from torchinfo import summary

class DoubleConv(nn.Module):
    def __init__(self, C_in, C_out, padding):
        super().__init__()
        self.conv = nn.Sequential(
                nn.Conv2d(C_in, C_out, 3, 1, padding, bias = False), # Bias = False since we use batchnorm
                nn.BatchNorm2d(C_out),
                nn.ReLU(inplace=True),
                nn.Conv2d(C_out, C_out, 3, 1, padding, bias = False), # Padding is set to 0 since we use overlapping-windows- 
                nn.BatchNorm2d(C_out),                          
                nn.ReLU(inplace=True)
                )
                                                                # -important when working with medical imaging or satellite image mosaics, to avoid edge artifacts
    def forward(self,x):
        return self.conv(x)

class UNET(nn.Module):
    def __init__(self, C_in=3, C_out=3, features = [64, 128, 256, 512], padding=0):
        super().__init__()
        self.ups = nn.ModuleList()
        self.downs = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)

        # Encoder parts
        for feature in features:
            self.downs.append(DoubleConv(C_in, feature, padding))
            C_in = feature
        
        # Decoder parts
        for feature in reversed(features):
            self.ups.append(
                    nn.ConvTranspose2d(                             # 1. up-conv feature*2 -> feature 
                        feature*2, feature, kernel_size=2, stride=2
                        ) 
                    )
            self.ups.append(DoubleConv(feature*2, feature, padding))         # 2. skip connection + feature = feature*2 -> feature
    
        # Bottleneck
        self.bottleneck = DoubleConv(features[-1], features[-1]*2, padding)
        # Final conv into segmentation maps
        self.final_conv = nn.Conv2d(features[0], C_out, kernel_size=1)
        # Softmax along features to get probabilities
        self.softm = nn.Softmax2d()


    def forward(self, x):
        skip_connections = []
        # Encoder
        print(">> Starting Encoder")
        print(f"x shape: {tuple(x.shape)}")
        for down in self.downs:
            x = down(x)
            print(f"double conv x shape: {tuple(x.shape)}")
            skip_connections.append(x)
            x = self.pool(x)
            print(f"maxpool x shape: {tuple(x.shape)}")

        # Bottleneck
        x = self.bottleneck(x)
        print(f"bottleneck x shape: {tuple(x.shape)}")
        skip_connections = skip_connections[::-1]
        
        # Decoder
        print(">> Starting Decoder stage")
        print(f"x shape: {tuple(x.shape)}")
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)  # Up-conv
            print(f"up-conv x shape: {tuple(x.shape)}")
            skip_connection = skip_connections[i//2] 
            if x.shape != skip_connection.shape:
                print(f"x-shape: {tuple(x.shape)} doesn't match skip connection: {tuple(skip_connection.shape)}")
                x_h,_ = x.shape[2:]
                skip_h,_= skip_connection.shape[2:]
                if x_h > skip_h:
                    x = TF.resize(x,size=skip_connection.shape[2:])
                else:
                    skip_connection = TF.resize(skip_connection,size=x.shape[2:])
                    
            
            concat_skip = torch.cat((skip_connection, x), dim=1) # (N, C, H, W) 
            x = self.ups[i+1](concat_skip) # X = g(f(X) + X)
            print(f"concat + double conv x shape: {tuple(x.shape)}")
        x = self.final_conv(x)
        return self.softm(x)
        
if __name__ == "__main__":
    x = torch.rand(3,3,572,572) # torch.rand samples from uniform dist [0,1) torch.randn samples from normal dist N(0,1)
    PADDING = 0
    model = UNET(C_in = 3, C_out=3, padding = PADDING)
    print(f"input shape: {tuple(x.shape)}")
    pred = model(x)

    summary(model, (3,3,572,572))
    print(f"output shape: {tuple(pred.shape)}")

