import os
from torch.optim import Adam
from utils.networkHelper import *
from noisePredictModels.Unet.UNet import Unet
from utils.trainNetworkHelper import SimpleDiffusionTrainer
from diffusionModels.simpleDiffusion.simpleDiffusion import DiffusionModel
from utils import dataread


data_root_path = './dataset'
data_num = 1

image_size = 128
channels = 1
batch_size = 6
timesteps = 200
epoches = 200
device = "cuda" if torch.cuda.is_available() else "cpu"

imagenet_data = dataread.Dataset(data_root_path, data_num, image_size, augment_horizontal_flip = False, convert_image_to = None)
data_loader = torch.utils.data.DataLoader(imagenet_data,
                                          batch_size=batch_size,
                                          shuffle=True,
                                          num_workers=0)

dim_mults = (1, 2, 4,)
denoise_model = Unet(
    dim=image_size,
    channels=channels,
    dim_mults=dim_mults
)

schedule_name = "linear_beta_schedule"
model = DiffusionModel(schedule_name=schedule_name,
                      timesteps=timesteps,
                      beta_start=0.00115,
                      beta_end=0.031,
                      denoise_model=denoise_model).to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

root_path = "./saved_train_models"
setting = "imageSize{}_channels{}_dimMults{}_timeSteps{}_scheduleName{}".format(image_size, channels, dim_mults,
                                                                                timesteps, schedule_name)
saved_path = os.path.join(root_path, setting)
if not os.path.exists(saved_path):
    os.makedirs(saved_path)

Trainer = SimpleDiffusionTrainer(epoches=epoches,
                                 mode='train',
                                 train_loader=data_loader,
                                 optimizer=optimizer,
                                 device=device,
                                 timesteps=timesteps)
model = Trainer(model, model_save_path=saved_path)

# 验证和处理数据请使用以下的代码并注释上面的代码，请确保验证的数据大小与模型参数一致.
# 需要验证的待处理数据存放在./dataset目录下，采用mat格式，待处理数据变量名为data。每一份数据按照正整数编号, 也可以更改这部分的代码读取其他类型数据。

