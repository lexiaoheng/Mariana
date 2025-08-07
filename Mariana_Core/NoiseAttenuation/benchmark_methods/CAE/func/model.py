import torch.nn as nn


class CAE_model(nn.Module):
    def __init__(self, input_c, hidden_c, bias = True):
        super(CAE_model, self).__init__()

        self.encoder = nn.Sequential(
            # in: 512 512

            nn.Conv2d(input_c, hidden_c, 4, 2, 1, bias=bias), # 256 256
            # nn.Tanh(),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_c, hidden_c * 2, 4, 2, 1, bias=bias), # 128 128
            # nn.Tanh(),
            # nn.BatchNorm2d(ndf * 2),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_c * 2, hidden_c * 4, 4, 2, 1, bias=bias), # 64 64
            # nn.Tanh(),
            # nn.BatchNorm2d(ndf * 4),
            # nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden_c * 4, hidden_c * 4, 4, 2, 1, bias=bias), # 32 32
            # nn.Tanh(),
            # nn.LeakyReLU(0.2, inplace=True),
        )

        self.decoder = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(hidden_c * 4, hidden_c * 4, 4, 2, 1, bias=bias),
            # nn.BatchNorm2d(ngf * 8),
            # nn.Tanh(),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*8) x 4 x 4
            nn.ConvTranspose2d(hidden_c * 4, hidden_c * 2, 4, 2, 1, bias=bias),
            # nn.BatchNorm2d(ngf * 4),
            # nn.Tanh(),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*4) x 8 x 8
            nn.ConvTranspose2d(hidden_c * 2, hidden_c, 4, 2, 1, bias=bias),
            # nn.BatchNorm2d(ngf * 2),
            # nn.Tanh(),
            # nn.LeakyReLU(0.2, inplace=True),
            # state size. (ngf*2) x 16 x 16
            nn.ConvTranspose2d(hidden_c, input_c, 4, 2, 1, bias=bias),
            # nn.Tanh()
            # nn.Sigmoid()

        )


    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
