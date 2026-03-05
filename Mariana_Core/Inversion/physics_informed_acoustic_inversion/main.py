import matplotlib.pyplot as plt
import os

from torch import optim
from tqdm import tqdm
from func.utils import *
from torch.utils.data import DataLoader
from func.utils import ImpedanceDataset1D

# 1. load data

data_name = 'Marmousi' # switch: Marmousi Overthrust SEAM
wave_name = 'ricker'
weighting = 'equal' # equal or UW
data_dic = np.load(('./data/' + data_name + '_' + wave_name + '_20Hz.npy'), allow_pickle=True).item()
seismic = data_dic['seismic']
impedance = data_dic['impedance'] / 1000

random_seed = 1234 # or 2026
torch.manual_seed(random_seed)
torch.cuda.manual_seed(random_seed)
np.random.seed(random_seed)
os.environ['PYTHONHASHSEED'] = str(random_seed)

## 2. build train set (synthetic data)
## 2.1 parameter

[h, w] = impedance.shape
batch_size = 6
epoch = 1000
lr = 0.004
wavelet_len = 301

## 2.2 build train set

mask = np.zeros([1,w])
if data_name == 'SEAM':
    mask[:,80::170] = 1 # SEAM 10 well logs
    # mask[:, 100::450] = 1 # 4 well logs
    # mask[:, 120::300] = 1 # 6 well logs
    # mask[:,350::190] = 1  # 8 well logs
elif data_name == 'Marmousi':
    mask[:, 165::110] = 1  # Marmousi 14 well logs
    # mask[:, 330::350] = 1  # 4 well logs
    # mask[:, 320::240] = 1  # 6 well logs
    # mask[:, 120::210] = 1 # 8 well logs
else:
    mask[:,90::100]=1 # Overthrust 10 well logs
    # mask[:, 100::250] = 1 # 4 well logs
    # mask[:, 300::150] = 1  # 6 well logs
    # mask[:, 120::120] = 1 # 8 well logs

impedance_log = impedance * mask
seismic, impedance_log, log_mean, log_std, seismic_mean, seismic_std = normal(seismic, impedance_log)
seismic_dataset1D = SeismicDataset1D(seismic)
seismic_dataloader1D = DataLoader(seismic_dataset1D, batch_size=batch_size, shuffle=True)
impedance_dataset1D = ImpedanceDataset1D(seismic, impedance_log, mask)
impedance_dataloader1D = DataLoader(impedance_dataset1D, batch_size=batch_size, shuffle=True)

## 3. train model

from func.model import Model
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

scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=int(epoch / 4), gamma=0.5)

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

snr, ssim, r2, mae, mse = metric(out, impedance)
print('SNR: %.4f , SSIM: %.4f , R2: %.4f, MAE: %.4f, MSE: %.4f ' % (snr, ssim, r2, mae, mse))

# visualization
if data_name == 'Marmousi':
    min = 2
    max = 12
elif data_name == 'Overthrust':
    min = 2
    max = 18
else:
    min = 2
    max = 12
plt.subplot(2,2,1)
plt.plot(out_wavelet)
plt.title('extracted wavelet')
plt.grid(True)
plt.subplot(2,2,2)
plt.imshow(impedance*(1-mask)+mask*10,'jet', vmin=min, vmax=max,aspect='auto')
plt.title('ground truth and wells')
plt.subplot(2,2,3)
plt.imshow(out,'jet', vmin=min, vmax=max,aspect='auto')
plt.title('predicted impedance')
plt.subplot(2,2,4)
plt.imshow(np.abs(impedance-out),'gist_yarg', vmin=0, vmax=4,aspect='auto')
plt.title('absolute residuals')
plt.show()

np.save('./output/' + data_name + '_' + wave_name + '_proposed.npy', arr = {'impedance': out, 'wave_out': out_wavelet})
# np.save('./output/loss/' + data_name + '_' + wave_name + '.npy', arr = {'loss_i': loss_i, 'loss_p': loss_p, 'loss_c': loss_c})





