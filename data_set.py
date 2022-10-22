# Dataset and Dataloader
import numpy as np
import pandas as pd
import torch
import torchvision
from torch.utils.data import Dataset, DataLoader
import math

# Download data
import os
import requests

# --> use dataset and dataloader to load data
'''
epoch = 1 forwards and backwards pass of all training samples

batch_size = number of training samples in one forward/backward pass

number of iterations = number of passes, each pass using [batch_size] number of samples

e.g. 100 samples, batch_size=20 --> 100/20 = 5 iterations for 1 epoch
'''

class WineDataset(Dataset):
    def __init__(self):
        super(WineDataset).__init__()
        # data loading
        xy = np.loadtxt('wine.csv', delimiter=',', dtype=np.float32, skiprows=1)
        self.x = torch.from_numpy(xy[:, 1:])
        self.y =torch.from_numpy(xy[:, [0]]) # n_samples, 1
        self.n_samples = xy.shape[0]        
        
    def __getitem__(self, index):
        # dataset[0]
        return self.x[index], self.y[index]
    
    def __len__(self):
        # len(dataset)
        return self.n_samples
    

def download_data():
    # download data
    url = "https://github.com/python-engineer/pytorchTutorial/raw/master/data/wine/wine.csv"
    r = requests.get(url, allow_redirects=True)
    open('wine.csv', 'wb').write(r.content)
    
    # file download check
    headers = requests.head(url).headers
    downloadable = 'attachment' in headers.get('Content-Disposition', '') # what is this?
    print(downloadable) # false...?
    
    # file type
    print(r.headers.get('content-type'))


# training loop
def train(dataset, batch_size):
    num_epochs = 2
    total_samples = len(dataset)
    n_iterations = math.ceil(total_samples / batch_size)
    print(total_samples, n_iterations)
    
    for epoch in range(num_epochs):
        for i, (inputs, labels) in enumerate(dataloader):
            # forward, backward, update
            if (i+1) % 5 == 0:
                print(f'epoch {epoch+1}/{num_epochs}, step {i+1}/{n_iterations}, inputs {inputs.shape}')
                

if __name__ == '__main__':
    if 'wine.csv' not in os.listdir():
        download_data()
    dataset = WineDataset()
    # first_data = dataset[0]
    # features, labels = first_data
    # print(features, labels)
    
    dataloader = DataLoader(dataset=dataset, batch_size = 4, shuffle=True, num_workers=2)
    
    dataiter = iter(dataloader) # iterator
    data = dataiter.next() # next batch
    print(len(dataloader)) # 178/4 = 45 (dataset size / batch size)
    print(len(dataset)) # 178
    # features, labels = data
    # print(features, labels)
    
    train(dataset, batch_size=4)
    
    # Also, you can get datas from famous datasets for example:
    # MNIST, CIFAR10, COCO, etc.
    torchvision.datasets.MNIST(root='data', download=True)
    # download=True --> download data if it doesn't exist
    