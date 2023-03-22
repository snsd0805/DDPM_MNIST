import torch
import torch.nn as nn
from torchinfo import summary
import math
from positionEncoding import PositionEncode

class DoubleConv(nn.Module):
    '''
        Have 2 convolutional layers, and we have to define the activation function in the last layer

        ( the output size will be same as the input size )

        Inputs:
            x: feature map, (b, in_dim, h, w)
        Outputs:
            x: feature map, (b, out_dim, h, w)
        Args:
            in_dim (int): input feature map's channel
            out_dim (int): output feature map's channel
            last_activation(nn.Module): the last layer's activation function, like nn.ReLU(), nn.Tanh()
    '''
    def __init__(self, in_dim, out_dim, last_activation):
        super(DoubleConv, self).__init__()
        self.conv1 = nn.Conv2d(in_dim, out_dim, 3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(out_dim, out_dim, 3, stride=1, padding=1)
        self.relu  = nn.ReLU()
        self.last_activation = last_activation
    
    def forward(self, x):
        x = self.relu(self.conv1(x))                                        # (b, out_dim, h, w)
        x = self.last_activation(self.conv2(x))                             # (b, out_dim, h, w)
        return x

class DownSampling(nn.Module):
    '''
        Unet used it to down sampling the picture
        
        Inputs:
            x: feature maps, (b, in_dim, h, w)
            time_emb: time_embedding, (b, time_emb_dim)
        Outputs:
            x: feature maps, (b, out_dim, h/2, w/2)
        Args:
            in_dim (int): input feature map's channel
            out_dim (int): output feature map's channel
            time_emb_dim (int): time embedding's dimension
    '''
    def __init__(self, in_dim, out_dim, time_emb_dim):
        super(DownSampling, self).__init__()
        self.pooling = nn.MaxPool2d(2, stride=2)
        self.conv1 = DoubleConv(in_dim, out_dim, nn.ReLU())
        self.time_linear = nn.Linear(time_emb_dim, out_dim)
    
    def forward(self, x, time_emb):
        x = self.pooling(x)                                                 # (b, in_dim, h/2, w/2)
        x = self.conv1(x)                                                   # (b, out_dim, h/2, w/2)
        b, c, h, w = x.shape

        time_emb = self.time_linear(time_emb)                               # (b, out_dim)
        time_emb = time_emb[:, :, None, None].repeat(1, 1, h, w)            # (b, out_dim, h/2, w/2)

        return x + time_emb                                                 # (b, out_dim, h/2, w/2)

class UpSampling(nn.Module):
    '''
        Inputs:
            x: feature maps, (b, in_dim, h, w)
            skip_x: feature maps, (b, in_dim/2, h*2, w*2)
            time_emb: time_embedding, (b, time_emb_dim)
        Outputs:
            x: feature maps, (b, out_dim, h*2, w*2)
        Args:
            in_dim (int): input feature map's channel
            out_dim (int): output feature map's channel
            time_emb_dim (int): time embedding's dimension
    '''

    def __init__(self, in_dim, out_dim, time_emb_dim):
        super(UpSampling, self).__init__()
        self.trans_conv = nn.ConvTranspose2d(in_dim, in_dim//2, kernel_size=2, stride=2)
        self.time_linear = nn.Linear(128, out_dim)
        self.conv = DoubleConv(in_dim, out_dim, nn.ReLU())
    
    def forward(self, x, skip_x, time_emb):
        x = self.trans_conv(x)                                              # (b, in_dim/2, h*2, w*2)
        x = torch.cat([x, skip_x], dim=1)                                   # (b, in_dim,   h*2, w*2)
        x = self.conv(x)                                                    # (b, out_dim,  h*2, w*2)
        b, c, h, w = x.shape

        time_emb = self.time_linear(time_emb)                               # (b, out_dim)
        time_emb = time_emb[:, :, None, None].repeat(1, 1, h, w)            # (b, out_dim,  h*2, w*2)
        return x + time_emb

class Unet(nn.Module):
    '''
        Unet module, predict the noise

        Inputs:
            x: x_t feature mamps, (b, c, h, w)
            time_seq: A longtensor means this x is x_t, In this module, it will transform time_seq to position embedding. (b, )
        Outputs:
            out: predicted noise (b, c, h, w)
        Args:
            time_emb_dim (int): dimention when position encoding
            device (nn.device): for PositionEncode layer, means the data puts on what device
    '''
    def __init__(self, time_emb_dim, device):
        super(Unet, self).__init__()
        self.in1 = DoubleConv(1, 32, nn.ReLU())
        self.down1 = DownSampling(32, 64, time_emb_dim)
        self.down2 = DownSampling(64, 128, time_emb_dim)
        self.latent1 = DoubleConv(128, 256, nn.ReLU())
        self.latent2 = DoubleConv(256, 256, nn.ReLU())
        self.latent3 = DoubleConv(256, 128, nn.ReLU())
        self.up1 = UpSampling(128, 64, time_emb_dim)
        self.up2 = UpSampling(64, 32, time_emb_dim)
        # self.out = DoubleConv(32, 1, nn.Tanh())
        self.out1 = nn.Conv2d(32, 32, 3, padding=1)
        self.out2 = nn.Conv2d(32,  1, 3, padding=1)
        self.relu = nn.ReLU()

        self.dropout05 = nn.Dropout(0.2)

        self.time_embedding = PositionEncode(time_emb_dim, device)
    
    def forward(self, x, time_seq): 
        time_emb = self.time_embedding(time_seq)                            # (b, time_emb_dim)

        l1 = self.in1(x)                                                    # (b, 32, 28, 28)
        l1 = self.dropout05(l1)
        l2 = self.down1(l1, time_emb)                                       # (b, 64, 14, 14)
        l2 = self.dropout05(l2)
        l3 = self.down2(l2, time_emb)                                       # (b,128,  7,  7)
        l3 = self.dropout05(l3)

        latent = self.latent1(l3)                                           # (b, 256, 7, 7)
        latent = self.dropout05(latent)
        latent = self.latent2(latent)                                       # (b, 256, 7, 7)
        latent = self.dropout05(latent)
        latent = self.latent3(latent)                                       # (b, 128, 7, 7)
        latent = self.dropout05(latent)

        l4 = self.up1(latent, l2, time_emb)                                 # (b, 64, 14, 14)
        l4 = self.dropout05(l4)
        l5 = self.up2(l4, l1, time_emb)                                     # (b, 32, 28, 28)
        l5 = self.dropout05(l5)
        out = self.relu(self.out1(l5))                                      # (b,  1, 28, 28)
        out = self.dropout05(out)
        out = self.out2(out)                                                # (b,  1, 28, 28)
        return out
