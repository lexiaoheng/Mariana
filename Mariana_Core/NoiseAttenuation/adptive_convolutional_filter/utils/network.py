import torch.nn as nn


class network(nn.Module):
    def __init__(self,n_chan,chan_embed=48,layer=4,bias=1):
        super(network, self).__init__()

        if bias==1:
            is_bias = True
        else:
            is_bias = False
        self.layers = nn.ModuleList()
        # self.act = nn.LeakyReLU(negative_slope=0.2, inplace=True)
        self.act = nn.Tanh()
        self.layers.append(nn.Conv2d(n_chan,chan_embed,3,padding=1,bias=is_bias))

        for i in range(layer-2):
            conv = nn.Conv2d(chan_embed, chan_embed, 3, padding = 1, bias=is_bias)
            self.layers.append(conv)
        self.layers.append(nn.Conv2d(chan_embed, n_chan, 1,bias=is_bias))

    def forward(self, x):
        le = len(self.layers)
        for index, layer in enumerate(self.layers):
            if index<le-1:
                x = self.act(layer(x))
            else:
                x = layer(x)

        return x

