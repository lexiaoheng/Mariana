import numpy as np
from skimage.metrics import structural_similarity as ssim


def evaluate(label, output, method='MAE'):
    if method == 'MAE':
        ab = np.abs(label - output)
        return np.mean(ab)

    if method == 'MSE':
        s_abs = (label - output) ** 2
        return np.mean(s_abs)

    if method == 'NMSE':
        mse = np.mean((label - output) ** 2)
        sigma2 = np.mean((label - np.mean(label)) ** 2)
        return mse / sigma2

    if method == 'SNR':
        return 10 * np.log10(np.sum(label ** 2) / np.sum(((label - output) ** 2)))

    if method == 'PCC':
        label_mean = np.mean(label)
        output_mean = np.mean(output)
        n = np.sum((label - label_mean) * (output - output_mean))
        d_x = np.sum((label - label_mean) ** 2)
        d_y = np.sum((output - output_mean) ** 2)
        d = (d_x * d_y) ** 0.5
        return n / d

    if method == 'R2':
        return 1 - np.sum((label - output) ** 2) / np.sum((label-np.mean(label)) ** 2)

    if method == 'SSIM':
        return ssim((output-np.min(output))/(np.max(output)-np.min(output)), (label-np.min(label))/(np.max(label)-np.min(label)), channel_axis = False, data_range=1)

    return print('Unknown method, avaluable methods include: \'MAE\', \'MSE\', \'NMSE\', \'SNR\', \'R2\', \'SSIM\', \'PCC\'.')

