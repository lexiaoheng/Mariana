import torch
import torch.nn.functional as F
import torch.nn as nn
from Mariana_Core.utils import noise_level_estimate

from tqdm import tqdm


def pair_downsampler(data):
    #data has shape B C H W
    c = data.shape[1]

    filter1 = torch.FloatTensor([[[[0 ,0.5],[0.5, 0]]]]).to(data.device)
    filter1 = filter1.repeat(c,1, 1, 1)

    filter2 = torch.FloatTensor([[[[0.5 ,0],[0, 0.5]]]]).to(data.device)
    filter2 = filter2.repeat(c,1, 1, 1)

    output1 = F.conv2d(data, filter1, stride=2, groups=c)
    output2 = F.conv2d(data, filter2, stride=2, groups=c)

    return output1, output2


def downsample_process(data, mode = 'downsample'):
    #data has shape B C H W
    if mode == 'downsample':
        b, c, h, w=data.shape
        filter1 = torch.FloatTensor([[[[1, 0], [0, 0]]]]).to(data.device)
        filter1 = filter1.repeat(c, 1, 1, 1)

        filter2 = torch.FloatTensor([[[[0, 1],[0, 0]]]]).to(data.device)
        filter2 = filter2.repeat(c, 1, 1, 1)

        filter3 = torch.FloatTensor([[[[0, 0], [1, 0]]]]).to(data.device)
        filter3 = filter3.repeat(c, 1, 1, 1)

        filter4 = torch.FloatTensor([[[[0, 0], [0, 1]]]]).to(data.device)
        filter4 = filter4.repeat(c, 1, 1, 1)

        output1 = F.conv2d(data, filter1, stride=2, groups=c)
        output2 = F.conv2d(data, filter2, stride=2, groups=c)
        output3 = F.conv2d(data, filter3, stride=2, groups=c)
        output4 = F.conv2d(data, filter4, stride=2, groups=c)

        return torch.stack([output1, output2, output3, output4], dim=0).reshape(b*4, c, int(h/2), int(w/2))

    else:

        [b, c, h, w]=data.shape
        data = data.reshape(4, int(b/4), c, h, w)
        output1 = data[0, :, :, :, :].reshape(int(b/4), c, h, w)
        output2 = data[1, :, :, :, :].reshape(int(b/4), c, h, w)
        output3 = data[2, :, :, :, :].reshape(int(b/4), c, h, w)
        output4 = data[3, :, :, :, :].reshape(int(b/4), c, h, w)

        output = torch.zeros([int(b / 4), c, h * 2, w * 2]).to(data.device)
        output[:, :, 0::2, 0::2] = output1
        output[:, :, 0::2, 1::2] = output2
        output[:, :, 1::2, 0::2] = output3
        output[:, :, 1::2, 1::2] = output4

        return output

def mse(gt: torch.Tensor, pred:torch.Tensor)-> torch.Tensor:
    loss = torch.nn.MSELoss()
    return loss(gt,pred)


def loss_func(noisy_img, model):
    noisy1, noisy2 = pair_downsampler(noisy_img)
    sigma = noise_level_estimate(noisy_img)

    if sigma is not None:

        pred1 = noisy1 - model(noisy1)
        pred2 = noisy2 - model(noisy2)

        loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))

        noisy_denoised = noisy_img - model(noisy_img)
        denoised1, denoised2 = pair_downsampler(noisy_denoised)

        loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))

        loss_std = abs(model(noisy_img).var()-sigma)

        loss = (loss_cons + loss_res) * 1 / 2 + 1 / 2 * loss_std

    else:

        pred1 = noisy1 - model(noisy1)
        pred2 = noisy2 - model(noisy2)

        loss_res = 1 / 2 * (mse(noisy1, pred2) + mse(noisy2, pred1))

        noisy_denoised = noisy_img - model(noisy_img)
        denoised1, denoised2 = pair_downsampler(noisy_denoised)

        loss_cons = 1 / 2 * (mse(pred1, denoised1) + mse(pred2, denoised2))
        loss = loss_cons + loss_res

    return loss


def train_single_epoch(model, optimizer, noisy_img):

    loss = loss_func(noisy_img, model)
    out = loss.item()
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    return out


def train(model, noisy_img, optimizer, scheduler, max_epoch):
    loss = []
    with tqdm(total=max_epoch) as t:
        for epoch in range(int(max_epoch)):
            loss_value = train_single_epoch(model, optimizer, noisy_img)
            scheduler.step()
            loss.append(loss_value)

            t.set_postfix(loss=loss_value)
            t.update(1)

    return loss


def denoise(model, noisy_img):

    model.eval()
    pred = noisy_img - model(noisy_img)

    return pred


