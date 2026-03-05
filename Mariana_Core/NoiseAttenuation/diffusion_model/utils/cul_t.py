from Mariana_Core.utils.noise_level_estimate import noise_level_estimate
import torch
import numpy as np


def cul_t(data):
    data_for_est = torch.tensor(data).unsqueeze(0).unsqueeze(0)
    estimation = 0
    h = data_for_est.shape[2]
    w = data_for_est.shape[3]
    while h > 9 and w > 9:
        sigma2 = int(noise_level_estimate(data_for_est))
        estimation = sigma2 if sigma2 > estimation else estimation

        data_for_est = data_for_est[:, :, 0::2, 0::2]
        h = data_for_est.shape[2]
        w = data_for_est.shape[3]

    beta = np.linspace(0.00115, 0.031, 200)
    alpha = 1 - beta
    sigma_y = np.var(data)
    sigma_2 = estimation
    overline_alpha_t_field = (sigma_y - sigma_2) / (sigma_2 * 1.0853 - sigma_2 + sigma_y)
    overline_alpha = np.cumprod(alpha)

    return np.argmin(abs(overline_alpha - overline_alpha_t_field))

