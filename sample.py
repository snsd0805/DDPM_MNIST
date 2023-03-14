import torch
import matplotlib.pyplot as plt
from ddpm import DDPM
from unet import Unet

BATCH_SIZE = 256
ITERATION = 500
TIME_EMB_DIM = 128
DEVICE = torch.device('cuda')

model = Unet(TIME_EMB_DIM, DEVICE).to(DEVICE)
ddpm = DDPM(BATCH_SIZE, ITERATION, 1e-4, 2e-2, DEVICE)

model.load_state_dict(torch.load('unet.pth'))

x_t = ddpm.sample(model, 256)
for index, pic in enumerate(x_t):
    p = pic.to('cpu').permute(1, 2, 0)
    plt.imshow(p, cmap='gray', vmin=0, vmax=255)
    plt.savefig("output/{}.png".format(index))

    