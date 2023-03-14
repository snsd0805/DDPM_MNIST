import torch
import torch.nn as nn
import matplotlib.pyplot as plt

class DDPM(nn.Module):
    '''
        Denoising Diffussion Probabilistic Model
        
        Inputs:

        Args:
            batch_size (int): batch_size, for generate time_seq, etc.
            iteration (int): max time_seq
            beta_min, beta_max (float): for beta scheduling
            device (nn.Device)
    '''
    def __init__(self, batch_size, iteration, beta_min, beta_max, device):
        super(DDPM, self).__init__()
        self.batch_size = batch_size
        self.iteration = iteration
        self.device = device

        self.beta = torch.linspace(beta_min, beta_max, steps=iteration).to(self.device)         # (iteration)
        self.alpha = (1 - self.beta).to(self.device)                                            # (iteration)
        self.overline_alpha = torch.cumprod(self.alpha, dim=0)

    def get_time_seq(self, length):
        '''
            Get random time sequence for each picture in the batch

            Inputs:
                length (int): size of sequence
            Outputs:
                time_seq: rand int from 0 to ITERATION
        '''
        return torch.randint(0, self.iteration, (length,) ).to(self.device)
    
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
        mu = torch.sqrt(self.overline_alpha[time_seq])[:, None, None, None].repeat(1, c, w, h)              # (b, c, w, h)
        mu = mu * x_0                                                                                       # (b, c, w, h)
        sigma = torch.sqrt(1-self.overline_alpha[time_seq])[:, None, None, None].repeat(1, c, w, h)         # (b, c, w, h)
        epsilon = torch.randn_like(x_0).to(self.device)                                                     # (b, c, w, h)

        return mu + sigma * epsilon, epsilon                                                                # (b, c, w, h)
    
    def sample(self, model, generate_iteration_pic=False, n=None):
        '''
            Inputs:
                model (nn.Module): Unet instance
                generate_iteration_pic (bool): whether generate 10 pic on different denoising time
                n (int, default=self.batch_size): want to sample n pictures
            Outputs:
                x_0 (nn.Tensor): (n, c, h, w)
        '''
        if n == None:
            n = self.batch_size
        c, h, w = 1, 28, 28
        model.eval()
        with torch.no_grad():
            x_t = torch.randn((n, c, h, w)).to(self.device)             # (n, c, h, w)
            for i in reversed(range(self.iteration)):
                time_seq = (torch.ones(n) * i).long().to(self.device)       # (n, )
                predict_noise = model(x_t, time_seq)                        # (n, c, h, w)

                first_term = 1/(torch.sqrt(self.alpha[time_seq]))           # (n, )
                second_term = (1-self.alpha[time_seq]) / (torch.sqrt(1-self.overline_alpha[time_seq]))

                first_term = first_term[:, None, None, None].repeat(1, c, h, w)
                second_term = second_term[:, None, None, None].repeat(1, c, h, w)

                beta = self.beta[time_seq][:, None, None, None].repeat(1, c, h, w)

                if i!= 0:
                    z = torch.randn((n, c, h, w)).to(self.device)
                else:
                    z = torch.zeros((n, c, h, w)).to(self.device)

                x_t = first_term * (x_t-(second_term * predict_noise)) - z * beta

                # generate 10 pic on the different denoising times
                if generate_iteration_pic:
                    if i % (self.iteration/10) == 0:
                        p = x_t[0].cpu()
                        p = ( p.clamp(-1, 1) + 1 ) / 2
                        p = p * 255
                        p = p.permute(1, 2, 0)
                        plt.imshow(p, vmin=0, vmax=255, cmap='gray')
                        plt.savefig("output/iter_{}.png".format(i))

        x_t = ( x_t.clamp(-1, 1) + 1 ) / 2
        x_t = x_t * 255
        return x_t