import torch
import numpy as np
from skimage.metrics import structural_similarity as ssim
from torch.utils.data import Dataset


class ImpedanceDataset1D(Dataset):

    def __init__(self, seismic, impedance):
        self.seismic = seismic
        self.model = impedance
        # self.loc_location = np.squeeze(log_location)
        self.trace_indices = np.array(np.nonzero(np.sum(abs(impedance), axis=0)))[0, :]
        # self.trace_indices = np.array(np.nonzero(np.squeeze(log_location)))
        print('Locations of well logs:', self.trace_indices)

    def __getitem__(self, index):

        trace_index = self.trace_indices[index]

        x = torch.tensor(self.seismic[: ,trace_index], dtype=torch.float).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
        y = torch.tensor(self.model[:, trace_index], dtype=torch.float).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        return x, y

    def __len__(self):
        return len(self.trace_indices)


class SeismicDataset1D(Dataset):

    def __init__(self, seismic):
        self.seismic = seismic

        self.h, self.w = self.seismic.shape
        self.trace_indices = np.linspace(0, self.w - 1, round(self.w / 20))

        print('Numbers of seismic traces:', self.w)
        print('Length of each seismic trace:', self.h)

    def __getitem__(self, index):

        trace_index = round(self.trace_indices[index])
        x = torch.tensor(self.seismic[:,trace_index], dtype=torch.float).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

        return x

    def __len__(self):
        return len(self.trace_indices)


def metric(data1, data2):

    snr = 10*np.log10(np.sum(data2**2)/np.sum((data1-data2)**2))
    sim = ssim((data1-np.min(data1))/(np.max(data1)-np.min(data1)), (data2-np.min(data2))/(np.max(data2)-np.min(data2)), multichannel = False, data_range=1)
    r2 = 1 - np.sum((data2-data1) ** 2) / np.sum((data2-np.mean(data2)) ** 2)

    data1 = (data1 - np.mean(data1)) / np.std(data1)
    data2 = (data2 - np.mean(data2)) / np.std(data2)

    mse = np.mean((data1-data2)**2)
    mae = np.mean(np.abs(data1-data2))

    return snr, sim, r2, mae, mse


def normal(seismic, impedance_log):

    # exact impedance
    log = impedance_log[:, [not np.all(impedance_log[:,i]==0) for i in range(impedance_log.shape[1])]]
    mask = np.zeros(impedance_log.shape)
    mask[:, [not np.all(impedance_log[:, i] == 0) for i in range(impedance_log.shape[1])]] = 1

    log_mean = np.mean(log)
    log_std = np.std(log)
    seismic_mean = np.mean(seismic)
    seismic_std = np.std(seismic)
    impedance_log = ((impedance_log-log_mean)/log_std) * mask
    seismic = (seismic-seismic_mean)/seismic_std

    return seismic, impedance_log, log_mean, log_std, seismic_mean, seismic_std