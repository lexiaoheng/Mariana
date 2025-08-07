from torch import optim
from tqdm import tqdm
from .func.utils import *
from torch.utils.data import DataLoader
from .func.utils import ImpedanceDataset1D
from .func.model import Model


def EIF(seismic, well_log):

    [_, w] = seismic.shape
    batch_size_seismic = 16
    batch_size_inversion = 6
    epoch_seismic = 30
    epoch_inversion = 1000
    lr_seismic = 0.001
    lr_inversion = 0.01

    seismic, well_log, log_mean, log_std = normal(seismic, well_log)

    seismic_dataset1D = SeismicDataset1D(seismic)
    seismic_dataloader1D = DataLoader(seismic_dataset1D,batch_size=batch_size_seismic, shuffle=True)

    impedance_dataset1D = ImpedanceDataset1D(seismic, well_log)
    impedance_dataloader1D = DataLoader(impedance_dataset1D,batch_size=batch_size_inversion, shuffle=True)

    model = Model(input_channels=1, L=seismic.shape[0], mode='2D').to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    model.train()
    loss_func = torch.nn.MSELoss()
    optimizer = optim.Adam(model.parameters(),lr_seismic, weight_decay=1e-4)

    mode = 'encoder'
    print('Training encoder')
    with tqdm(total=epoch_seismic) as t:
        for i in range(epoch_seismic):
            for x, n in seismic_dataloader1D:

                re, n_out = model(x, mode)

                loss1 = loss_func(x, re)
                loss2 = loss_func(n, n_out)
                loss = loss1 + loss2

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            t.set_postfix(loss=loss.item(), reconstruct_loss=loss1.item(), domain_predict_loss=loss2.item())
            t.update(1)

    mode = 'inversion'
    print('Fine-tuning inverter')
    optimizer = optim.Adam(model.parameters(),lr_inversion, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epoch_inversion / 4), gamma=0.5)
    model.partly_freeze()
    with tqdm(total=epoch_inversion) as t:
        for i in range(epoch_inversion):
            for x, y in impedance_dataloader1D:

                out = model(x, mode)
                loss = loss_func(y, out)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

            t.set_postfix(loss=loss.item())
            t.update(1)

    mode = 'validation'
    output = seismic
    print(mode)
    model.eval()

    with tqdm(total=w) as t:
        for i in range(w):
            input = torch.tensor(seismic[:, i], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            x, _, _ = model(input, mode)
            output[:, i] = x.cpu().detach().squeeze(0).squeeze(0).numpy()
            
            t.update(1)

    return output * log_std + log_mean





