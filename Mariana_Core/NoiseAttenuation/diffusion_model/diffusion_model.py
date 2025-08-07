import os
from Mariana_Core.NoiseAttenuation.diffusion_model.utils.networkHelper import *
from Mariana_Core.NoiseAttenuation.diffusion_model.noisePredictModels.Unet.UNet import Unet
from Mariana_Core.NoiseAttenuation.diffusion_model.utils.trainNetworkHelper import SimpleDiffusionTrainer
from Mariana_Core.NoiseAttenuation.diffusion_model.diffusionModels.simpleDiffusion.simpleDiffusion import DiffusionModel
from Mariana_Core.NoiseAttenuation.diffusion_model.utils.cul_t import cul_t
from Mariana_Core.NoiseAttenuation.diffusion_model.utils.norm import norm


def DPM(data, ddim=True):

    if len(data.shape) == 2:
        t_seq = cul_t(data)
        data, field_mu, field_sigma2 = norm(data, 5.0297 * 0.00001, 1.0853, t_seq, 'norm')
    else:
        num = data.shape[2]
        t_seq = np.zeros([num])
        field_mu = np.zeros([num])
        field_sigma2 = np.zeros([num])
        for i in range(num):
            t_seq[i] = cul_t(data[:, :, i])
            data[:, :, i], field_mu[i], field_sigma2[i] = norm(data[:, :, i], 5.0297 * 0.00001, 1.0853, int(t_seq[i]), 'norm')


    image_size = 128
    channels = 1
    timesteps = 200
    dim_mults = (1, 2, 4,)
    schedule_name = "linear_beta_schedule"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    denoise_model = Unet(
        dim=image_size,
        channels=channels,
        dim_mults=dim_mults
    )
    model = DiffusionModel(schedule_name=schedule_name,
                          timesteps=timesteps,
                          beta_start=0.00115,
                          beta_end=0.031,
                          denoise_model=denoise_model).to(device)

    best_model_path = './Mariana_core/NoiseAttenuation/diffusion_model/Pretrained_model.pth'
    if os.path.exists(best_model_path):
        model.load_state_dict(torch.load(best_model_path))

        DPM = SimpleDiffusionTrainer(mode='validation',
                                         device=device,
                                         timesteps=timesteps)

        out = DPM(model, noisy=data, t_seq=t_seq, ddim=ddim)
        if len(out.shape) == 2:
            out = norm(out, field_mu, field_sigma2, t_seq, 'reverse')
        else:
            for i in range(num):
                out[:, :, i] = norm(out[:, :, i], field_mu[i], field_sigma2[i], int(t_seq[i]), 'reverse')

        return out

    else:
        print('Can`t find pretrained model.')
