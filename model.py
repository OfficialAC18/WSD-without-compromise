import torch
import torch.nn as nn
import torch.nn.functional as F

class Encoder(nn.Module):
    '''
    Encoder for the Beta-VAE model in the paper.
    Architecture:
    Input - 64x64x3
    Conv1 - 4x4 kernel, 32 filters, stride 2
    ReLU
    Conv2 - 4x4 kernel, 32 filters, stride 2
    ReLU
    Conv3 - 4x4 kernel, 64 filters, stride 2
    ReLU
    Conv4 - 4x4 kernel, 64 filters, stride 2
    ReLU
    FC - 256 units
    F2 2x10 units
    '''
    
    def __init__(self):
        self.conv1 = nn.Conv2d(3, 32, 4, 2,)
        self.conv2 = nn.Conv2d(32, 32, 4, 2)
        self.conv3 = nn.Conv2d(32, 64, 4, 2)
        self.conv4 = nn.Conv2d(64, 64, 4, 2)
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.Linear(256,10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = x.view(-1, 256)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class Decoder(nn.Module):
    '''
    Decoder for the Beta-VAE model in the paper.
    Architecture:
    Input - R^10,
    FC - 256 units
    ReLU
    FC - 4x4x64 units
    ReLU
    UpConv1 - 4x4 kernel, 64 filters, stride 2
    ReLU
    UpConv2 - 4x4 kernel, 32 filters, stride 2
    ReLU
    UpConv3 - 4x4 kernel, 32 filters, stride 2
    ReLU
    UpConv4 - 4x4 kernel, 3 filters, stride 2
    '''

    def __init__ (self):
        self.fc1 = nn.LazyLinear(256)
        self.fc2 = nn.LazyLinear(4*4*64)
        self.upconv1 = nn.ConvTranspose2d(64, 64, 4, 2)
        self.upconv2 = nn.ConvTranspose2d(64, 32, 4, 2)
        self.upconv3 = nn.ConvTranspose2d(32, 32, 4, 2)
        self.upconv4 = nn.ConvTranspose2d(32, 3, 4, 2)   


    def forward(self,x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = x.view(-1, 64, 4, 4)
        x = F.relu(self.upconv1(x))
        x = F.relu(self.upconv2(x))
        x = F.relu(self.upconv3(x))
        x = F.relu(self.upconv4(x))
        return x 

