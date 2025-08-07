import torch
from tqdm import tqdm
import numpy as np
from scipy.interpolate import CubicSpline


def mask_generator(data):

    mask = torch.randn_like(data)
    mask[mask >= 0] = 1
    mask[mask < 0] = 0

    return mask


def pre_interpolation(data, mask):

    b, c, h, w = data.shape
    device = data.device
    data = data.squeeze(0).squeeze(0).cpu().numpy()
    mask = mask.squeeze(0).squeeze(0).cpu().numpy()
    refine = np.zeros([h, int(np.sum(mask[0, :]))])
    location = np.zeros([int(np.sum(mask[0, :])), 1])

    n = 0
    out = np.zeros([h, w])
    for i in range(w):
        if mask[0, i] == 1:
            refine[:, n] = data[:, i]
            location[n, 0] = i
            n = n + 1

    o=np.linspace(0, w, w, endpoint=False)

    for i in range(h):
        cs = CubicSpline(location.flatten(), refine[i, :].flatten())
        out[i, :]=cs(o.flatten())

    # out = data + (1 - mask) * out

    return torch.tensor(out).unsqueeze(0).unsqueeze(0).float().to(device)


def model_train(data, model, optimizer, scheduler, loss_func, epoch=8000, mode='zs-scl', missing_mask=None):

    loss1_all = []
    loss2_all = []
    loss3_all = []

    if mode == 'zs-scl':

        std = torch.sqrt(((data - data.sum().item() / missing_mask.sum().item()) ** 2).sum() / missing_mask.sum().item()).item()
        aver = (data.sum().item() / missing_mask.sum().item())
        data = (data - aver) / std

        data = pre_interpolation(data, missing_mask)
        with tqdm(total=epoch) as t:
            for i in range(epoch):

                out1 = model(data)
                loss1 = loss_func(out1 * missing_mask, data * missing_mask)

                mask = mask_generator(out1)
                out2 = model(mask * out1)

                loss2 = loss_func(out2 * missing_mask, data * missing_mask)
                loss3 = loss_func(out1, out2)

                loss = loss1 + loss2 + loss3

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                t.set_postfix(loss=loss.item(), loss1=loss1.item(), loss2=loss2.item(), loss3=loss3.item())
            
                loss1_all.append(loss1.item())
                loss2_all.append(loss2.item())
                loss3_all.append(loss3.item())
                
                t.update(1)

        data = model(data)
        out = ((data - data.mean().item()) / (torch.std(data).item())) * std + aver

    if mode == 'traditional':

        std = torch.sqrt(
            ((data - data.sum().item() / missing_mask.sum().item()) ** 2).sum() / missing_mask.sum().item()).item()
        aver = (data.sum().item() / missing_mask.sum().item())
        data = (data - aver) / std

        data = pre_interpolation(data, missing_mask)
        with tqdm(total=epoch) as t:
            for i in range(epoch):
                out = model(data)
                loss1 = loss_func(out * missing_mask, data * missing_mask)
                optimizer.zero_grad()
                loss1.backward()
                optimizer.step()
                scheduler.step()
                t.set_postfix(loss=loss1.item())
                loss1_all.append(loss1.item())
                t.update(1)
        data = model(data)
        out = ((data - data.mean().item()) / (torch.std(data).item())) * std + aver

    return model, out, loss1_all, loss2_all, loss3_all
