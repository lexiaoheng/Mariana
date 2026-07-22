from torch import optim
from tqdm import tqdm

from .func.utils import *
from torch.utils.data import DataLoader
from .func.utils import ImpedanceDataset1D


def RDSCL(seismic, well_log):

    ## 1. parameters
    batch_size = 6
    epoch = 1000
    lr = 0.002
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    ## 2. dataset
    seismic, well_log, log_mean, log_std, seismic_mean, seismic_std = normal(seismic, well_log)
    [h, length] = seismic.shape
    print('Size of seismic data: %d * %d' % (h, length))
    seismic_dataset1D = SeismicDataset1D(seismic, device=device)
    seismic_dataloader1D = DataLoader(seismic_dataset1D, batch_size=batch_size*4, shuffle=True)
    impedance_dataset1D = ImpedanceDataset1D(seismic=seismic, impedance=well_log, device=device)
    impedance_dataloader1D = DataLoader(impedance_dataset1D, batch_size=batch_size, shuffle=True)

    ## 3. workflow

    ## 3.1. instantiation of model
    from .func.model import Model
    model = Model().to(device)
    e_num, i_num = model_summary(model)
    print('Total trainable parameters:',e_num + i_num)

    ## 3.2. training
    model.train()
    loss_func = torch.nn.MSELoss()
    optimizer = optim.AdamW(model.parameters(), lr, weight_decay=1e-2)

    with tqdm(total=epoch) as t:
        for i in range(epoch):

            seismic_iter = iter(seismic_dataloader1D)
            for x, y in impedance_dataloader1D:

                out = model(x)
                loss1 = loss_func(y, out)
                x_nolabel = next(seismic_iter)

                # warm up with 50 epoches
                if i<(epoch*0.05):
                    loss = loss1
                    t.set_postfix(loss=loss.item(), loss1=loss1.item(), loss2=None)
                else:
                    y = y * log_std + log_mean
                    x = x * seismic_std + seismic_mean
                    x_wavelet = wavelet_deconvolution(y, x)

                    out_nolabel = model(x_nolabel)
                    x_nolabel = x_nolabel * seismic_std + seismic_mean
                    out_nolabel = out_nolabel * log_std + log_mean
                    x_nolabel_wavelet = wavelet_deconvolution(out_nolabel, x_nolabel)

                    loss2 = (loss_func(x_nolabel, forward_synthetic(x_wavelet, out_nolabel)) + loss_func(x, forward_synthetic(x_nolabel_wavelet, y))) / 2
                    loss = loss1 + loss2
                    t.set_postfix(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item())

                optimizer.zero_grad()
                loss.backward()
                # gradient clip with L2 norm
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1)
                optimizer.step()

            t.update(1)

    ## 3.3. inversion on whole profile
    model.eval()
    input_seismic = torch.tensor(seismic, dtype=torch.float).unsqueeze(0).permute(2, 0, 1).to(device)
    with tqdm(total=1) as t:
        out = model(input_seismic)
        t.update(1)
    out = out.permute(1, 2, 0).cpu().detach().squeeze(0).numpy()
    out = out * log_std + log_mean

    return out