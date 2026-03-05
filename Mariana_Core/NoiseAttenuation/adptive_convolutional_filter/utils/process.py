import torch.optim as optim
from .network import *
from .utils import *


def process(noisy_data, training_param=None, network_param=None, model=None):
    if network_param is None:
        network_param = {'chan_embed': 48, 'layers': 4, 'bias': 1}
    if training_param is None:
        training_param = {'device':'cuda','max_epoch': 100, 'lr': 0.001, 'step_size': 10, 'gamma': 0.5}

    noisy_data = noisy_data.to(training_param['device'])

    max_epoch = training_param['max_epoch']  # training epochs
    lr = training_param['lr']  # learning rate
    step_size = training_param['step_size']  # number of epochs at which learning rate decays
    gamma = training_param['gamma']  # factor by which learning rate decays

    if model is None:
        model = network(noisy_data.shape[1],chan_embed=network_param['chan_embed'],layer=network_param['layers'], bias=network_param['bias'])
        model = model.to(training_param['device'])

    optimizer = optim.Adam(model.parameters(), lr=lr)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=step_size, gamma=gamma)
    loss = train(model, noisy_data, optimizer, scheduler, max_epoch)
    out = denoise(model, noisy_data)

    return  model, out, loss