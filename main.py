import torch
import torch.nn as nn



if __name__ == '__main__':
    print('main code')
    print(torch.cuda.is_available())
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    # number of gpu
    print(torch.cuda.device_count())
    # current gpu
    print(torch.cuda.current_device())
    # gpu name
    print(torch.cuda.get_device_name(0))