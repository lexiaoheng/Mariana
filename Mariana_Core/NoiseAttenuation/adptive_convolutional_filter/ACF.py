from .utils.process import process
from .utils.utils import *
import numpy as np


def ACF(data):

    # pre-process data

    ma = np.max(data)
    mi = np.min(data)
    input_data = (data - mi) / (ma - mi)

    # parameters

    chan_embed = 16
    layers = 3
    max_epoch = 500
    bias = 0
    orientation = 'noise'
    network_param = {'chan_embed': chan_embed, 'layers': layers, 'bias': bias}
    training_param = {'device': 'cuda', 'max_epoch': max_epoch, 'lr': 0.001, 'step_size': max_epoch / 3, 'gamma': 0.5}

    # process

    input_data = torch.Tensor(input_data).unsqueeze(0).unsqueeze(0)
    [_, output1, _] = process(input_data, training_param=training_param, network_param=network_param,
                                    model=None)

    input2 = downsample_process(output1, 'downsample').detach()
    [_, output2, _] = process(input2, training_param=training_param, network_param=network_param,
                                    model=None)
    input3 = downsample_process(output2, 'reverse').detach()

    [_, final, _] = process(input3, training_param=training_param, network_param=network_param,
                                    model=None)

    return final.cpu().detach().squeeze(0).squeeze(0).numpy() * (ma- mi) + mi



