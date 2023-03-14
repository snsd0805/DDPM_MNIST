import torch
import torch.nn as nn
from torchvision.datasets import MNIST
import torchvision
from torch.utils.data import DataLoader, Dataset
import matplotlib.pyplot as plt
from tqdm import tqdm
from ddpm import DDPM
from unet import Unet
import configparser

def getMnistLoader(config):
    '''
        Get MNIST dataset's loader

        Inputs:
            config (configparser.ConfigParser)
        Outputs:
            loader (nn.utils.data.DataLoader)
    '''
    BATCH_SIZE      =   int(config['unet']['batch_size'])

    transform = torchvision.transforms.Compose([
        torchvision.transforms.ToTensor()
    ])

    data = MNIST("./data", train=True, download=True, transform=transform)
    loader = DataLoader(data, batch_size=BATCH_SIZE, shuffle=True)
    return loader

def train(config):
    '''
        Start Unet Training

        Inputs:
            config (configparser.ConfigParser)
        Outputs:
            None
    '''
    BATCH_SIZE      =   int(config['unet']['batch_size'])
    ITERATION       =   int(config['ddpm']['iteration'])
    TIME_EMB_DIM    =   int(config['unet']['time_emb_dim'])
    DEVICE          =   torch.device(config['unet']['device'])
    EPOCH_NUM       =   int(config['unet']['epoch_num'])
    LEARNING_RATE   =   float(config['unet']['learning_rate'])

    # training
    model = Unet(TIME_EMB_DIM, DEVICE).to(DEVICE)
    ddpm = DDPM(BATCH_SIZE, ITERATION, 1e-4, 2e-2, DEVICE)

    criterion = nn.MSELoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    min_loss = 99

    for epoch in range(EPOCH_NUM):
        loss_sum = 0
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
        
        print("Epoch {}/{}: With lr={}, batch_size={}, iteration={}. The best loss: {} - loss: {}".format(epoch, EPOCH_NUM, LEARNING_RATE, BATCH_SIZE, ITERATION, min_loss, loss_sum/len(loader)))
        if loss_sum/len(loader) < min_loss:
            min_loss = loss_sum/len(loader)
            print("save model: the best loss is {}".format(min_loss))
            torch.save(model.state_dict(), 'unet.pth')

if __name__ == '__main__':
    # read config file
    config = configparser.ConfigParser()
    config.read('training.ini')

    # start training
    loader = getMnistLoader(config)
    train(config)