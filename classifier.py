import torch
import torch.nn as nn
from unet import DownSampling, DoubleConv
from positionEncoding import PositionEncode

class Classfier(nn.Module):
    '''
        Args:
            time_emb_dim (int): dimention when position encoding
            device (nn.device): for PositionEncode layer, means the data puts on what device
        Inputs:
            x: feature maps, (b, c, h, w)
            time_seq: A longtensor means this x is x_t, In this module, it will transform time_seq to position embedding. (b, ) 
        Outputs:
            p: probability (b, 10)
    '''
    def __init__(self, time_emb_dim, device):
        super(Classfier, self).__init__()
        self.conv1 = DoubleConv(1, 32, nn.ReLU())
        self.conv2 = DownSampling(32, 64, time_emb_dim)
        self.conv3 = DownSampling(64, 128, time_emb_dim)
        self.conv4 = DownSampling(128, 256, time_emb_dim)
        self.dense1 = nn.Linear(256*3*3, 512)
        self.dense2 = nn.Linear(512, 128)
        self.dense3 = nn.Linear(128, 10)
        self.pooling = nn.AvgPool2d(2, stride=2)
        self.relu = nn .ReLU()
        self.dropout = nn.Dropout(0.3)

        self.time_embedding = PositionEncode(time_emb_dim, device)
    
    def forward(self, x, time_seq):
        time_emb = self.time_embedding(time_seq)         # (b, time_emb_dim)

        x = self.conv1(x,)                               # b, 32, 28, 28
        x = self.conv2(x, time_emb)                      # b, 64, 14, 14
        x = self.dropout(x)
        x = self.conv3(x, time_emb)                      # b, 128, 7,  7
        x = self.dropout(x)
        x = self.conv4(x, time_emb)                      # b, 256, 3, 3
        x = self.dropout(x)

        x = x.reshape((x.shape[0], -1))                  # b, 2304
        x = self.relu(self.dense1(x))                   # b, 512
        x = self.dropout(x)
        x = self.relu(self.dense2(x))                   # b, 128
        x = self.dropout(x)
        x = self.dense3(x)                              # b, 10
        return x
