import torch
import torch.nn as nn
from unet import Unet
import matplotlib.pyplot as plt

class DDPM(nn.Module):
    '''
        Denoising Diffussion Probabilistic Model
        
        Inputs:

        Args:
            batch_size (int): batch_size, for generate time_seq, etc.
            iteration (int): max time_seq
            beta_min, beta_max (float): for beta scheduling
            time_emb_dim (int): for Unet's PositionEncode layer
            device (nn.Device)
    '''
    def __init__(self, batch_size, iteration, beta_min, beta_max, time_emb_dim, device):
        super(DDPM, self).__init__()
        self.batch_size = batch_size
        self.iteration = iteration
        self.device = device
        self.unet = Unet(time_emb_dim, device)
        self.time_emb_dim = time_emb_dim

        self.beta = torch.linspace(beta_min, beta_max, steps=iteration)             # (iteration)
        self.alpha = 1 - self.beta                                                  # (iteration)
        self.overline_alpha = torch.cumprod(self.alpha, dim=0)

    def get_time_seq(self):
        '''
            Get random time sequence for each picture in the batch

            Inputs:
                None
            Outputs:
                time_seq: rand int from 0 to ITERATION
        '''
        return torch.randint(0, self.iteration, (self.batch_size,) )
    
    def get_x_t(self, x_0, time_seq):
        '''
            Input pictures then return noised pictures

            Inputs:
                x_0: pictures (b, c, w, h)
                time_seq: times apply on each pictures (b, )
            Outputs:
                x_t: noised pictures (b, c, w, h)
        '''
        b, c, w, h = x_0.shape
        mu = torch.sqrt(self.overline_alpha[time_seq])[:, None, None, None].repeat(1, c, w, h)
        mu = mu * x_0

        sigma = torch.sqrt(1-self.overline_alpha[time_seq])[:, None, None, None].repeat(1, c, w, h)
        epsilon = torch.randn_like(x_0)

        return mu + sigma * epsilon
