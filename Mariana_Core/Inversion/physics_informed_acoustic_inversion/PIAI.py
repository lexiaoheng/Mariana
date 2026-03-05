import matplotlib.pyplot as plt
import os

from torch import optim
from tqdm import tqdm
from .func.utils import *
from .func.model import Model
from torch.utils.data import DataLoader
from .func.utils import ImpedanceDataset1D

# 1. load data


def PIAI(seismic, well_log, weighting='Equal', wavelet_len=301):

    [h, w] = seismic.shape
    batch_size = 6
    epoch = 1000
    lr = 0.004
    # wavelet_len = 301

    seismic, impedance_log, log_mean, log_std, seismic_mean, seismic_std = normal(seismic, well_log)
    seismic_dataset1D = SeismicDataset1D(seismic)
    seismic_dataloader1D = DataLoader(seismic_dataset1D, batch_size=batch_size, shuffle=True)

    impedance_dataset1D = ImpedanceDataset1D(seismic, impedance_log)
    impedance_dataloader1D = DataLoader(impedance_dataset1D, batch_size=batch_size, shuffle=True)

    model = Model(input_channels=1, L=seismic.shape[0], wavelet_len=wavelet_len).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    e_num, i_num, w_num = model_summary(model)
    print('total parameters:',e_num + i_num + w_num)
    print('Encoder parameters: %d, Inverter parameters: %d, Wavelet exactor parameters: %d' % (e_num, i_num, w_num))

    model.train()
    loss_func = torch.nn.MSELoss()

    if weighting == 'UW':
        uw = AutomaticWeightedLoss(3)
        optimizer = optim.AdamW([
        {'params': model.parameters(), 'weight_decay': 1e-2},
        {'params': uw.parameters(), 'weight_decay': 0}
        ], lr)
    else:
        optimizer = optim.AdamW(model.parameters(),lr, weight_decay=1e-2)

    # scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epoch / 4), gamma=0.5)

    loss_p = []
    loss_c = []
    loss_i = []
    with tqdm(total=epoch) as t:
        for i in range(epoch):

            seismic_iter = iter(seismic_dataloader1D)
            for x, y in impedance_dataloader1D:

                out, wavelet = model(x)

                loss1 = loss_func(y, out)

                y = y * log_std + log_mean
                out = out * log_std + log_mean
                loss2 = loss_func(x, forward_synthetic(wavelet, y))
                loss_I = loss1 + loss2
                loss_P = loss_func(x, forward_synthetic(wavelet, out))

                x_nolabel = next(seismic_iter)
                out_nolabel, wavelet2 = model(x_nolabel)
                out_nolabel = out_nolabel*log_std+log_mean
                loss_C = (loss_func(x_nolabel, forward_synthetic(wavelet.detach(), out_nolabel)) + loss_func(x, forward_synthetic(wavelet2, y))) / 2

                if weighting == 'UW':
                    loss = uw(loss_I, loss_P, loss_C)
                else:
                    loss = loss_P + loss_C + loss_I

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            loss_i.append(loss_I.item())
            loss_p.append(loss_P.item())
            loss_c.append(loss_C.item())

            t.set_postfix(loss=loss.item(), Independent_Learning=loss_I.item(), Physics_Informed_Learning=loss_P.item(), Cross_Learning=loss_C.item())
            t.update(1)

    model.eval()
    out = seismic
    out_wavelet = np.zeros([wavelet_len, w])
    with tqdm(total=w) as t:
        for i in range(w):
            input_data = torch.tensor(seismic[:, i], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            temp_out, wavelet = model(input_data)
            out[:, i] = temp_out.cpu().detach().squeeze(0).squeeze(0).numpy()
            out_wavelet[:, i] = wavelet[0].cpu().detach().squeeze(0).squeeze(0).numpy()
            t.update(1)

    out = out * log_std + log_mean
    out_wavelet = np.mean(out_wavelet, axis=1) * 2 * seismic_std + seismic_mean

    return out, out_wavelet







