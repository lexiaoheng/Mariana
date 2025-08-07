from tqdm import tqdm
from Mariana_Core.NoiseAttenuation.diffusion_model.utils.networkHelper import *
from Mariana_Core.NoiseAttenuation.diffusion_model.diffusionModels.simpleDiffusion.varianceSchedule import VarianceSchedule
import torch.nn.functional as F


class DiffusionModel(nn.Module):
    def __init__(self,
                 schedule_name="linear_beta_schedule",
                 timesteps=200,
                 beta_start=0.0001,
                 beta_end=0.02,
                 denoise_model=None):
        super(DiffusionModel, self).__init__()

        self.denoise_model = denoise_model

        # 方差生成
        variance_schedule_func = VarianceSchedule(schedule_name=schedule_name, beta_start=beta_start, beta_end=beta_end)
        self.timesteps = timesteps
        self.betas = variance_schedule_func(timesteps)
        # define alphas
        self.alphas = 1. - self.betas
        self.alphas_cumprod = torch.cumprod(self.alphas, dim=0)
        self.alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.sqrt_recip_alphas = torch.sqrt(1.0 / self.alphas)

        # calculations for diffusion q(x_t | x_{t-1}) and others
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1. - self.alphas_cumprod)

        # calculations for posterior q(x_{t-1} | x_t, x_0)
        # 这里用的不是简化后的方差而是算出来的
        self.posterior_variance = self.betas * (1. - self.alphas_cumprod_prev) / (1. - self.alphas_cumprod)

    def q_sample(self, x_start, t, noise=None):
        # forward diffusion (using the nice property)
        # x_t  = sqrt(alphas_cumprod)*x_0 + sqrt(1 - alphas_cumprod)*z_t
        if noise is None:
            noise = torch.randn_like(x_start)

        sqrt_alphas_cumprod_t = extract(self.sqrt_alphas_cumprod, t, x_start.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x_start.shape
        )

        return sqrt_alphas_cumprod_t * x_start + sqrt_one_minus_alphas_cumprod_t * noise

    def compute_loss(self, x_start, t, noise=None, loss_type="l1"):
        if noise is None:
            noise = torch.randn_like(x_start)

        x_noisy = self.q_sample(x_start=x_start, t=t, noise=noise)
        predicted_noise = self.denoise_model(x_noisy, t)

        if loss_type == 'l1':
            loss = F.l1_loss(noise, predicted_noise)
        elif loss_type == 'l2':
            loss = F.mse_loss(noise, predicted_noise)
        elif loss_type == "huber":
            loss = F.smooth_l1_loss(noise, predicted_noise)
        else:
            raise NotImplementedError()

        return loss

    @torch.no_grad()
    def p_sample(self, x, t, t_index):
        betas_t = extract(self.betas, t, x.shape)
        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        )
        sqrt_recip_alphas_t = extract(self.sqrt_recip_alphas, t, x.shape)

        # Equation 11 in the paper
        # Use our model (noise predictor) to predict the mean
        model_mean = sqrt_recip_alphas_t * (
                x - betas_t * self.denoise_model(x, t) / sqrt_one_minus_alphas_cumprod_t
        )

        if t_index == 0:
            return model_mean
        else:
            posterior_variance_t = extract(self.posterior_variance, t, x.shape)
            noise = torch.randn_like(x)
            # Algorithm 2 line 4:
            return model_mean + torch.sqrt(posterior_variance_t) * noise

    @torch.no_grad()
    def p_sample_loop(self, image, t):
        device = next(self.denoise_model.parameters()).device

        b = 1
        img = image

        with tqdm(total=int(t), desc='sampling loop time step', position=1, leave=False) as tq:
            for i in reversed(range(0, t)):
                img = self.p_sample(img, torch.full((b,), i, device=device, dtype=torch.long), i)
                tq.update(1)

        return img

    @torch.no_grad()
    def sample(self,t=16, image=1):
        return self.p_sample_loop(image,t)

    @torch.no_grad()
    def fast_sample(self, x, t, t_pre):

        sqrt_one_minus_alphas_cumprod_t = extract(
            self.sqrt_one_minus_alphas_cumprod, t, x.shape
        ) # 根号下1-alpha_t累乘
        sqrt_one_minus_alphas_cumprod_t_pre = extract(
            self.sqrt_one_minus_alphas_cumprod, t_pre, x.shape
        )  # 根号下1-alpha_t_pre累乘

        alphas_cumprod_t_pre = extract(
            self.alphas_cumprod, t_pre, x.shape
        )

        alphas_cumprod_t = extract(
            self.alphas_cumprod, t, x.shape
        )

        # model_mean = (1-torch.pow(sqrt_one_minus_alphas_cumprod_t_pre,2)) * (x - sqrt_one_minus_alphas_cumprod_t * self.denoise_model(x, t)) / (1 - torch.pow(sqrt_one_minus_alphas_cumprod_t, 2)) + sqrt_one_minus_alphas_cumprod_t_pre * self.denoise_model(x, t)
        model_mean = alphas_cumprod_t_pre * (
                    x - sqrt_one_minus_alphas_cumprod_t * self.denoise_model(x, t)) / alphas_cumprod_t + sqrt_one_minus_alphas_cumprod_t_pre * self.denoise_model(x, t)

        return model_mean

    @torch.no_grad()
    def p_fast_sample_loop(self, image, t):
        device = next(self.denoise_model.parameters()).device

        b = 1
        img = image
        # imgs = []
        t_seq = ddim_t_seq(t)

        with tqdm(total=len(t_seq), desc='sampling loop time step', position=1, leave=False) as t:

            for i in reversed(range(0, len(t_seq))):

                if i > 0 :
                    img = self.fast_sample(img, torch.full((b,), t_seq[i], device=device, dtype=torch.long),torch.full((b,), t_seq[i-1], device=device, dtype=torch.long))
                    # imgs.append(img.cpu().numpy())

                elif i == 0 :
                    betas_0 = extract(self.betas, torch.full((1,), 0, device=device, dtype=torch.long), img.shape)
                    sqrt_one_minus_alphas_cumprod_0 = extract(
                        self.sqrt_one_minus_alphas_cumprod, torch.full((1,), 0, device=device, dtype=torch.long), img.shape
                    )
                    sqrt_recip_alphas_0 = extract(self.sqrt_recip_alphas, torch.full((1,), 0, device=device, dtype=torch.long), img.shape)

                    img = sqrt_recip_alphas_0 * (
                            img - betas_0 * self.denoise_model(img, torch.full((1,), 0, device=device, dtype=torch.long)) / sqrt_one_minus_alphas_cumprod_0
                    )
                t.update(1)

        return img

    def forward(self, mode, **kwargs):
        if mode == "train":
            # 先判断必须参数
            if "x_start" and "t" in kwargs.keys():
                # 接下来判断一些非必选参数
                if "loss_type" and "noise" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"],
                                             noise=kwargs["noise"], loss_type=kwargs["loss_type"])
                elif "loss_type" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], loss_type=kwargs["loss_type"])
                elif "noise" in kwargs.keys():
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"], noise=kwargs["noise"])
                else:
                    return self.compute_loss(x_start=kwargs["x_start"], t=kwargs["t"])

            else:
                raise ValueError("扩散模型在训练时必须传入参数x_start和t！")

        elif mode == "generate":
            if "x_start" in kwargs.keys():
                if "whether_ddim" in kwargs.keys():
                    if kwargs["whether_ddim"] == 1:
                        return self.p_fast_sample_loop(image=kwargs["x_start"],t=kwargs["t"])
                    else:
                        return self.p_sample_loop(image=kwargs["x_start"], t=kwargs["t"])
                else:
                    return self.p_sample_loop(image=kwargs["x_start"], t=kwargs["t"])

            else:
                raise ValueError("扩散模型在生成图片时必须传入image_size, batch_size, channels等三个参数")
        else:
            raise ValueError("mode参数必须从{train}和{generate}两种模式中选择")





