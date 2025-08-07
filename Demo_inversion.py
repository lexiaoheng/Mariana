import numpy as np
import os
import torch
import matplotlib.pyplot as plt

from Mariana_Core import Inversion, utils

# random seed
random_seed = 1234
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

# read data
data_dic = np.load('./Demo_data/testdata_for_inversion.npy', allow_pickle=True).item()
seismic = data_dic['seismic']
impedance_ground_truth = data_dic['impedance'] / 1000

# construct well logs
[h, w] = impedance_ground_truth.shape
mask = np.zeros([1, w])
mask[:, 170::110] = 1  # Marmousi 170::110
well_logs = impedance_ground_truth * mask

# use eif
out = Inversion.EIF(seismic, well_logs)
print('SNR: %.4f , SSIM: %.4f , R2: %.4f, MAE: %.4f, MSE: %.4f ' % (utils.evaluate(impedance_ground_truth, out, 'SNR'),
                                                                    utils.evaluate(impedance_ground_truth, out, 'SSIM'),
                                                                    utils.evaluate(impedance_ground_truth, out, 'R2'),
                                                                    utils.evaluate(impedance_ground_truth, out, 'MAE'),
                                                                    utils.evaluate(impedance_ground_truth, out, 'MSE')))

# visualization
plt.subplot(1, 3, 1)
plt.imshow(seismic, cmap='RdGy', aspect='auto')
plt.title('seismic data')

plt.subplot(1,3,2)
plt.imshow(impedance_ground_truth*(1-mask)+mask*20,'jet', vmin=2, vmax=18,aspect='auto')
plt.title('ground truth and wells')
plt.colorbar()

plt.subplot(1,3,3)
plt.imshow(out,'jet', vmin=2, vmax=18,aspect='auto')
plt.title('inverted results')
plt.colorbar()

plt.show()