import numpy as np


def norm(data, mu, sigma2, t, mode='norm'):
    beta = np.linspace(0.00115, 0.031, 200)
    alpha = 1 - beta
    overline_alpha = np.cumprod(alpha)

    if mode == 'norm':
        data_mu = np.mean(data)
        data_sigma2 = np.var(data)

        out = mu * np.sqrt(overline_alpha[t - 1]) + ((data - data_mu) / np.sqrt(data_sigma2)) * np.sqrt(
            overline_alpha[t - 1] * sigma2 + 1 - overline_alpha[t - 1])

        return out, data_mu, data_sigma2

    if mode == 'reverse':

        data_mu = np.mean(data)
        data_sigma2 = np.var(data)
        sigma2 = sigma2 - sigma2*(1 - overline_alpha[t - 1])/(1.0853*overline_alpha[t - 1]-overline_alpha[t - 1]+1)
        out = ((data - data_mu) / np.sqrt(data_sigma2)) * np.sqrt(sigma2) + mu

        return out

    return None
