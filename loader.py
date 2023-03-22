import torchvision
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader, Dataset

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
