from itertools import chain
from torch import optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from .func.utils import *
from .func.utils import ImpedanceDataset1D, SeismicDataset1D
from .func.model import inverse_model, forward_model


def DBDI(seismic, well_log):

    [_, w] = seismic.shape

    batch_size_inversion = 10
    epoch_inversion = 1000
    lr_inversion = 0.005

    seismic, impedance_log, log_mean, log_std, seismic_mean, seismic_std = normal(seismic, well_log)

    impedance_dataset1D = ImpedanceDataset1D(seismic, impedance_log)
    impedance_dataloader1D = DataLoader(impedance_dataset1D,batch_size=batch_size_inversion, shuffle = False)

    unlabeled_dataset1D = SeismicDataset1D(seismic)
    unlabeled_loader1D = DataLoader(unlabeled_dataset1D, batch_size=batch_size_inversion, shuffle = False)

    inverse = inverse_model().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
    forward = forward_model().to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))

    loss_func = torch.nn.L1Loss()
    optimizer = optim.AdamW(params=chain(inverse.parameters(),forward.parameters()),lr = lr_inversion, weight_decay=1e-4)

    print('Training')
    with tqdm(total=epoch_inversion) as t:
        for i in range(epoch_inversion):

            unlabeled_iter = iter(unlabeled_loader1D)
            for x, y in impedance_dataloader1D:

                out = inverse(x)
                loss1 = loss_func(y, out)

                re = forward(out)
                loss2 = loss_func(x, re)

                x_nolabel = next(unlabeled_iter)
                loss3 = loss_func(x_nolabel, forward(inverse(x_nolabel)))

                loss = 0.7 * loss1 + 0.2 * loss2 + 0.1 * loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                optimizer.step()

            t.set_postfix(loss=loss.item(), loss_supervised = loss1.item(), loss_forward = loss2.item(), loss_no_label = loss3.item())
            t.update(1)

    print('Validation')

    output = seismic
    inverse.eval()
    with tqdm(total=w) as t:
        for i in range(w):
            input = torch.tensor(seismic[:, i], dtype=torch.float).unsqueeze(0).unsqueeze(0).to(torch.device('cuda' if torch.cuda.is_available() else 'cpu'))
            x = inverse(input)
            output[:, i] = x.cpu().detach().squeeze(0).squeeze(0).numpy()
            t.update(1)

    output = output * log_std + log_mean

    return output


