import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from ddpm import DDPM
from unet import Unet

BATCH_SIZE = 512
ITERATION = 1500
TIME_EMB_DIM = 128
DEVICE = torch.device('cuda')
EPOCH_NUM = 3000
LEARNING_RATE = 1e-3

def getMnistLoader():

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    data = MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def train(loader, device, epoch_num, lr):
    model = Unet(TIME_EMB_DIM, DEVICE).to(device)
    ddpm = DDPM(BATCH_SIZE, ITERATION, 1e-4, 2e-2, device)

    criterion = nn.MSELoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=lr)

    for epoch in range(epoch_num):
        loss_sum = 0
        # progress = tqdm(total=len(loader))
        for x, y in loader:
            optimzer.zero_grad()

            x = x.to(DEVICE)
            time_seq = ddpm.get_time_seq(x.shape[0])
            x_t, noise = ddpm.get_x_t(x, time_seq)

            predict_noise = model(x_t, time_seq)
            loss = criterion(predict_noise, noise)

            loss_sum += loss.item()
            
            loss.backward()
            optimzer.step()
            # progress.update(1)
        torch.save(model.state_dict(), 'unet.pth')
        print("Epoch {}/{}: With lr={}, batch_size={}, iteration={}. loss: {}".format(epoch, EPOCH_NUM, LEARNING_RATE, BATCH_SIZE, ITERATION, loss_sum/len(loader)))

loader = getMnistLoader()
train(loader, DEVICE, EPOCH_NUM, LEARNING_RATE)