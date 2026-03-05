import torch.nn as nn


class CAE(nn.Module):
    def __init__(self, input_c, hidden_c, bias = True):
        super(CAE, self).__init__()

        self.encoder = nn.Sequential(
            nn.Conv2d(input_c, hidden_c, 4, 2, 1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(hidden_c, hidden_c * 2, 4, 2, 1, bias=bias),
            nn.Tanh(),
            nn.Conv2d(hidden_c * 2, hidden_c * 4, 6, 2, 2, bias=bias),
            nn.Tanh(),
            nn.Conv2d(hidden_c * 4, hidden_c * 8, 6, 2, 2, bias=bias),
            nn.Tanh(),
        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(hidden_c * 8, hidden_c * 4, 6, 2, 2, bias=bias),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_c * 4, hidden_c * 2, 6, 2, 2, bias=bias),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_c * 2, hidden_c, 4, 2, 1, bias=bias),
            nn.Tanh(),
            nn.ConvTranspose2d(hidden_c, input_c, 4, 2, 1, bias=bias),

        )

    def forward(self, x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x
