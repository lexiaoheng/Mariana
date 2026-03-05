import numpy as np
import torch
import torch.optim as optim
from .func.model import CAE_model
from tqdm import tqdm


def CAE(data, hidden_dim=4):

    # 1. pre-process data

    data_mean = np.mean(data)
    data = data - data_mean
    data = torch.tensor(data).unsqueeze(0).unsqueeze(0).float()

    # 2. train model
    # 2.1 parameters

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    max_epoch = 2000
    lr = 0.001

    input_data = data.to(device)
    model = CAE_model(1, hidden_dim).to(device)
    loss_func = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    # 2.2 training

    model.train()
    with tqdm(total=max_epoch) as t:
        for i in range(max_epoch):
            out = model(input_data)
            loss = loss_func(out, input_data)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            t.set_postfix(loss=loss.item())
            t.update(1)

    model.eval()
    out = model(input_data)

    return out.cpu().detach().squeeze(0).squeeze(0).numpy()



