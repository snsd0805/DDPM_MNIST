import torch
import torch.nn as nn
import math

class PositionEncode(nn.Module):
    '''
        Input a LongTensor time sequence, return position embedding

        Input:
            time_seq: shape of LongTensor (b, )
        Output:
            dim: shape of tensor (b, time_emb_dim)
        Args:
            time_emb_dim (int): output's dimension
            device (nn.device): data on what device 
    '''
    def __init__(self, time_emb_dim, device):
        super(PositionEncode, self).__init__()
        self.time_emb_dim = time_emb_dim
        
        self.base = torch.Tensor([ 1/math.pow(10000, (i//2)/self.time_emb_dim) for i in range(self.time_emb_dim) ]) # (d)
        self.base = self.base.to(device)

    def forward(self, time_seq):
        seq_len = len(time_seq)
        dim = self.base.reshape(1, self.time_emb_dim).repeat(seq_len, 1)    # (b, time_emb_dim)
        time_seq = time_seq[:, None].repeat(1, self.time_emb_dim)           # (b, time_emb_dim)
        ans = dim * time_seq                                                # (b, time_emb_dim)
        ans[:, 0::2] = torch.sin(ans[:, 0::2])
        ans[:, 1::2] = torch.cos(ans[:, 1::2])
        return ans
