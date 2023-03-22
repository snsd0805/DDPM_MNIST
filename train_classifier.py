import torch
import torch.nn as nn
import configparser
from loader import getMnistLoader
from classifier import Classfier
from ddpm import DDPM

def train(config):
    '''
        Start Classier Training

        Inputs:
            config (configparser.ConfigParser)
        Outputs:
            None
    '''
    BATCH_SIZE      =   int(config['classifier']['batch_size'])
    ITERATION       =   int(config['ddpm']['iteration'])
    TIME_EMB_DIM    =   int(config['classifier']['time_emb_dim'])
    DEVICE          =   torch.device(config['classifier']['device'])
    EPOCH_NUM       =   int(config['classifier']['epoch_num'])
    LEARNING_RATE   =   float(config['classifier']['learning_rate'])

    # training
    model = Classfier(TIME_EMB_DIM, DEVICE).to(DEVICE)
    ddpm = DDPM(BATCH_SIZE, ITERATION, 1e-4, 2e-2, DEVICE)

    criterion = nn.CrossEntropyLoss()
    optimzer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    min_loss = 99

    for epoch in range(EPOCH_NUM):
        loss_sum = 0
        acc_sum = 0
        data_count = 0
        for x, y in loader:
            optimzer.zero_grad()

            x = x.to(DEVICE)
            y = y.to(DEVICE)
            time_seq = ddpm.get_time_seq(x.shape[0])
            x_t, noise = ddpm.get_x_t(x, time_seq)

            p = model(x_t, time_seq)
            loss = criterion(p, y)

            loss_sum += loss.item()
            data_count += len(x)
            acc_sum += (p.argmax(1)==y).sum()
            
            loss.backward()
            optimzer.step()
        
        print("Epoch {}/{}: With lr={}, batch_size={}, iteration={}. The best loss: {} - loss: {}, acc: {}".format(epoch, EPOCH_NUM, LEARNING_RATE, BATCH_SIZE, ITERATION, min_loss, loss_sum/len(loader), acc_sum/data_count))
        if loss_sum/len(loader) < min_loss:
            min_loss = loss_sum/len(loader)
            print("save model: the best loss is {}".format(min_loss))
            torch.save(model.state_dict(), 'classifier.pth')

if __name__ == '__main__':
    # read config file
    config = configparser.ConfigParser()
    config.read('training.ini')

    # start training
    loader = getMnistLoader(config)
    train(config)