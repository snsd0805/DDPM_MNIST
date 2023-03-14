import torch
import matplotlib.pyplot as plt
from ddpm import DDPM
from unet import Unet
import sys
import os
import configparser

if __name__ == '__main__':
    if len(sys.argv) != 2:
        print("Usage: python sample.py [pic_num]")
        exit()
    
    # read config file
    config = configparser.ConfigParser()
    config.read('training.ini')

    BATCH_SIZE      =   int(config['unet']['batch_size'])
    ITERATION       =   int(config['ddpm']['iteration'])
    TIME_EMB_DIM    =   int(config['unet']['time_emb_dim'])
    DEVICE          =   torch.device(config['unet']['device'])

    # start sampling
    model = Unet(TIME_EMB_DIM, DEVICE).to(DEVICE)
    ddpm = DDPM(int(sys.argv[1]), ITERATION, 1e-4, 2e-2, DEVICE)

    model.load_state_dict(torch.load('unet.pth'))

    x_t = ddpm.sample(model)

    if not os.path.isdir('./output'):
        os.mkdir('./output')
    
    for index, pic in enumerate(x_t):
        p = pic.to('cpu').permute(1, 2, 0)
        plt.imshow(p, cmap='gray', vmin=0, vmax=255)
        plt.savefig("output/{}.png".format(index))