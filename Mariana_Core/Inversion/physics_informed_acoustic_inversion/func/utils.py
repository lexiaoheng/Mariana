import torch
import numpy as np
from torch.utils.data import Dataset
from torch.nn.functional import conv1d
import torch.nn as nn

class ImpedanceDataset1D(Dataset):

    def __init__(self, seismic, impedance):
        self.seismic = seismic
        self.model = impedance
        self.trace_indices = np.array(np.nonzero(np.sum(abs(impedance), axis=0)))[0, :]
        # self.trace_indices = np.array(np.nonzero(np.squeeze(log_location)))
        print('Locations of well logs:', self.trace_indices)

    def __getitem__(self, index):

        trace_index = self.trace_indices[index]

        x = torch.tensor(self.seismic[:,trace_index], dtype=torch.float).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        y = torch.tensor(self.model[:,trace_index], dtype=torch.float).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        return x, y

    def __len__(self):
        return len(self.trace_indices)


class SeismicDataset1D(Dataset):

    def __init__(self, seismic):

        self.seismic = seismic#[:, 0::5]

        self.h, self.w = self.seismic.shape
        print('Numbers of seismic traces:', seismic.shape[0])
        print('Length of each seismic trace:', seismic.shape[1])

    def __getitem__(self, index):

        x = torch.tensor(self.seismic[:,index], dtype=torch.float).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        # n = torch.tensor((index + 1) / self.w, dtype=torch.float).unsqueeze(0).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        return x

    def __len__(self):
        return self.w

def normal(seismic, impedance_log):

    # exact impedance
    mask = np.zeros(impedance_log.shape)
    mask[:, [not np.all(impedance_log[:, i] == 0) for i in range(impedance_log.shape[1])]] = 1

    log = impedance_log[:, [not np.all(impedance_log[:, i]==0) for i in range(impedance_log.shape[1])]]

    log_mean = np.mean(log)
    log_std = np.std(log)

    impedance_log = ((impedance_log-log_mean) / log_std) * mask
    seismic_mean = np.mean(seismic)
    seismic_std = np.std(seismic)
    seismic = (seismic - seismic_mean) / seismic_std

    return seismic, impedance_log, log_mean, log_std, seismic_mean, seismic_std


def model_summary(model: torch.nn.Module) -> (dict, list):

    Encoder_num = 0
    Inverter_num = 0
    Wavelet_num = 0

    for name, param in model.named_parameters():

        layer_name = name.split('.')[0]
        params_count = param.numel()

        if layer_name == 'encoder':
            Encoder_num += params_count
        elif layer_name == 'inverter':
            Inverter_num += params_count
        elif layer_name == 'wavelet_exactor':
            Wavelet_num += params_count

    return Encoder_num, Inverter_num, Wavelet_num


def forward_synthetic(wavelet, impedance):

    impedance_d = impedance[..., 1:] - impedance[..., :-1]
    impedance_a = (impedance[..., 1:] + impedance[..., :-1]) / 2
    wavelet = torch.mean(wavelet, dim=0, keepdim=True)

    r = (impedance_d / impedance_a)

    for i in range(impedance.shape[0]):
        tmp_synth = conv1d(r[[i], ...], wavelet, padding=int(wavelet.shape[-1] / 2 + 1))

        if i == 0:
            synth = tmp_synth
        else:
            synth = torch.cat((synth, tmp_synth), dim=0)

    _, _, h = impedance.shape
    synth = synth[..., 0:h]

    return synth


class AutomaticWeightedLoss(nn.Module):
    """
    Params：
        num: int，the number of loss
        x: multi-task loss
    """
    def __init__(self, num=3):
        super(AutomaticWeightedLoss, self).__init__()
        params = torch.ones(num, requires_grad=True)
        self.params = torch.nn.Parameter(params)

    def forward(self, *x):
        loss_sum = 0
        for i, loss in enumerate(x):
            loss_sum += 0.5 / (self.params[i] ** 2) * loss + torch.log(1 + self.params[i] ** 2)
        return loss_sum