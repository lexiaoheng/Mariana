import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset
from torch.nn.functional import conv1d
import torch.nn as nn

class ImpedanceDataset1D(Dataset):

    def __init__(self, seismic, impedance, device):
        self.device = device
        self.seismic = seismic
        self.model = impedance
        self.trace_indices = np.array(np.nonzero(np.sum(abs(impedance), axis=0)))[0, :]
        print('Locations of well logs:', self.trace_indices)

    def __getitem__(self, index):

        trace_index = self.trace_indices[index]

        x = torch.tensor(self.seismic[:,trace_index], dtype=torch.float32).unsqueeze(0).to(self.device)
        y = torch.tensor(self.model[:,trace_index], dtype=torch.float32).unsqueeze(0).to(self.device)

        return x, y

    def __len__(self):
        return len(self.trace_indices)


class SeismicDataset1D(Dataset):

    def __init__(self, seismic, device):

        self.seismic = seismic#[:, 0::2]
        self.device = device

        self.h, self.w = self.seismic.shape
        print('Numbers of seismic traces:', seismic.shape[1])
        print('Length of each seismic trace:', seismic.shape[0])

    def __getitem__(self, index):

        x = torch.tensor(self.seismic[:, index], dtype=torch.float).unsqueeze(0).to(self.device)

        return x

    def __len__(self):
        return self.w


def metric(data1, data2):

    snr = 10*np.log10(np.sum(data2**2)/np.sum((data1-data2)**2))
    sim = ssim((data1-np.min(data1))/(np.max(data1)-np.min(data1)), (data2-np.min(data2))/(np.max(data2)-np.min(data2)), channel_axis = False, data_range=1)
    r2 = 1 - np.sum((data1-data2) ** 2) / np.sum((data2-np.mean(data2)) ** 2)

    data1 = (data1 - np.mean(data1)) / np.std(data1)
    data2 = (data2 - np.mean(data2)) / np.std(data2)

    mse = np.mean((data1-data2)**2)
    mae = np.mean(np.abs(data1-data2))

    return snr, sim, r2, mae, mse


def normal(seismic, impedance_log):

    mask = np.zeros(impedance_log.shape)
    mask[:, [not np.all(impedance_log[:, i] == 0) for i in range(impedance_log.shape[1])]] = 1

    # exact impedance
    log = impedance_log[:, [not np.all(impedance_log[:,i]==0) for i in range(impedance_log.shape[1])]]

    log_mean = np.mean(log)
    log_std = np.std(log)

    impedance_log = ((impedance_log-log_mean) / log_std)
    seismic_mean = np.mean(seismic)
    seismic_std = np.std(seismic)
    seismic = (seismic - seismic_mean) / seismic_std

    return seismic, impedance_log*mask, log_mean, log_std, seismic_mean, seismic_std


def model_summary(model: torch.nn.Module) -> (dict, list):

    Encoder_num = 0
    Inverter_num = 0

    for name, param in model.named_parameters():

        layer_name = name.split('.')[0]
        params_count = param.numel()

        if layer_name == 'encoder':
            Encoder_num += params_count
        elif layer_name == 'inverter':
            Inverter_num += params_count

    return Encoder_num, Inverter_num


def forward_synthetic(wavelet, impedance):

    [B, _, L] = impedance.shape
    impedance_d = impedance[..., 1:] - impedance[..., :-1]
    impedance_a = (impedance[..., 1:] + impedance[..., :-1])

    r = impedance_d / impedance_a
    r_fft = torch.fft.rfft(r, n=L, dim=2, norm='backward')

    wavelet = torch.mean(wavelet, dim=0, keepdim=True)
    wave_fft = torch.fft.rfft(wavelet, n=L, dim=2, norm='backward')

    for i in range(B):

        tmp_synth = r_fft[i, :, :] * wave_fft
        if i == 0:
            synth = torch.fft.fftshift(torch.fft.irfft(tmp_synth, n=L, dim=2, norm='backward'))
        else:
            synth = torch.cat((synth, torch.fft.fftshift(torch.fft.irfft(tmp_synth, n=L, dim=2, norm='backward'))), dim=0)

    return synth


def wavelet_deconvolution(imp, seis, lam=1e-2):

    [_, _, L] = imp.shape
    imp_d = imp[..., 1:] - imp[..., :-1]
    imp_a = (imp[..., 1:] + imp[..., :-1])
    r = imp_d / imp_a

    r = torch.fft.rfft(r, n=L, dim=2, norm='backward')
    seis = torch.fft.rfft(seis, n=L, dim=2, norm='backward')

    d_sq = 2 - 2 * torch.cos(2 * torch.pi * torch.arange(r.shape[-1], device=r.device, dtype=r.real.dtype) / L)

    wavelet = torch.fft.fftshift(
        torch.fft.irfft( seis * torch.conj(r) / (r.abs() ** 2 + lam * d_sq), n=L, dim=2, norm='backward')
    ) # first order

    return wavelet

