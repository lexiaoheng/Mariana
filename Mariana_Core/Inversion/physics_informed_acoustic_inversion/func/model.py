import torch.nn as nn

gru_init_std = 0.000
conv_init_std = 0.01


class TemporalBlock1d(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.00):
        super(TemporalBlock1d, self).__init__()

        self.activation = nn.Tanh()

        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation, bias=True)

        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

        self.one_conv = nn.Conv1d(n_inputs, n_outputs, 1, bias=True)

        self.init_weights()

    def init_weights(self):

        self.conv1.weight.data.normal_(0, conv_init_std)
        self.conv2.weight.data.normal_(0, conv_init_std)
        self.one_conv.weight.data.normal_(0, conv_init_std)

    def freeze(self):

        self.conv1.weight.requires_grad = False
        self.conv1.bias.requires_grad = False
        self.conv2.weight.requires_grad = False
        self.conv2.bias.requires_grad = False

        self.one_conv.weight.requires_grad = False
        self.one_conv.bias.requires_grad = False

    def forward(self, x):

        out = self.conv1(x)
        out = self.activation(out)
        out = self.dropout1(out)

        out = self.conv2(out)
        out = self.activation(out)
        out = self.dropout2(out)

        res = self.one_conv(x)

        return self.activation(out + res)


class Encoder(nn.Module):

    def __init__(self, input_channels=1, num_channels=[16, 16, 16, 32], kernel_size=5, dropout=0.00):
        super(Encoder, self).__init__()

        tcn_layers = []
        self.num_levels = len(num_channels)

        for i in range(self.num_levels-1):

            dilation_size = 2
            in_channels = input_channels if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            tcn_layers.append(TemporalBlock1d(in_channels, out_channels, kernel_size, stride=1, dilation=dilation_size,
                                       padding=int((kernel_size-1) / 2 * dilation_size), dropout=dropout))

        self.layers = nn.Sequential(*tcn_layers)
        self.output = nn.Conv1d(in_channels=num_channels[-2], out_channels=num_channels[-1], kernel_size=3, padding=1, dilation=1)

    def freeze(self):

        for i in range(self.num_levels-1):
            self.layers[i].freeze()

        self.output.weight.requires_grad = False
        self.output.bias.requires_grad = False

    def forward(self, x):

        x = self.layers(x)
        x = self.output(x)

        return x


class Inverter(nn.Module):

    def __init__(self, input_channels=1, output_channels=1):
        super(Inverter, self).__init__()

        self.gru_out1 = nn.GRU(input_size=input_channels,
                               hidden_size=input_channels,
                               num_layers=3,
                               batch_first=True,
                               bidirectional=True)
        self.out = nn.Linear(in_features=input_channels * 2, out_features=output_channels)

    def forward(self, x):

        x, _ = self.gru_out1(x.transpose(-1, -2))
        x = self.out(x)
        x = x.transpose(-1, -2)

        return x


class Wavelet_Exactor(nn.Module):

    def __init__(self, input_channels=1, output_channels=1, wavelet_len=100, seis_len=1000):
        super(Wavelet_Exactor, self).__init__()

        self.gru1 = nn.GRU(input_size=input_channels,
                               hidden_size=input_channels,
                               num_layers=3,
                               batch_first=True,
                               bidirectional=True)
        self.out1 = nn.Linear(in_features=input_channels * 2, out_features=output_channels)
        self.out2 = nn.Linear(in_features=seis_len, out_features=wavelet_len)

    def freeze(self):

        for i in range(len(self.gru1.all_weights)):

            self.gru1.all_weights[i][0].requires_grad = False
            self.gru1.all_weights[i][1].requires_grad = False
            self.gru1.all_weights[i][2].requires_grad = False
            self.gru1.all_weights[i][3].requires_grad = False

        self.out1.weight.requires_grad = False
        self.out1.bias.requires_grad = False
        self.out2.weight.requires_grad = False
        self.out2.bias.requires_grad = False

    def forward(self, x):

        x, _ = self.gru1(x.transpose(-1, -2))
        x = self.out1(x)
        x = x.transpose(-1, -2)
        x = self.out2(x)

        return x

# class Reconstructor(nn.Module):
#
#     def __init__(self, input_channels=1, output_channels=1):
#         super(Reconstructor, self).__init__()
#
#         self.gru1 = nn.GRU(input_size=input_channels,
#                                hidden_size=input_channels,
#                                num_layers=3,
#                                batch_first=True,
#                                bidirectional=True)
#         self.out = nn.Linear(in_features=input_channels * 2, out_features=output_channels)
#
#     def freeze(self):
#
#         for i in range(len(self.gru1.all_weights)):
#
#             self.gru1.all_weights[i][0].requires_grad = False
#             self.gru1.all_weights[i][1].requires_grad = False
#             self.gru1.all_weights[i][2].requires_grad = False
#             self.gru1.all_weights[i][3].requires_grad = False
#
#         self.out.weight.requires_grad = False
#         self.out.bias.requires_grad = False
#
#     def forward(self, x):
#
#         x, _ = self.gru1(x.transpose(-1, -2))
#         x = self.out(x)
#         x = x.transpose(-1, -2)
#
#         return x
#
#
# class Liner(nn.Module):
#
#     def __init__(self, input_channels = 1, mode='2D', L = 1200):
#         super(Liner, self).__init__()
#
#         self.gru_n = nn.GRU(input_size=input_channels,
#                             hidden_size=1,
#                             num_layers=1,
#                             batch_first=True,
#                             bidirectional=True)
#
#         self.n1 = nn.Linear(in_features=2, out_features=1)
#         self.n2 = nn.Linear(in_features=L, out_features=1 if mode == '2D' else 2)
#
#     def freeze(self):
#
#         for i in range(len(self.gru_n.all_weights)):
#             self.gru_n.all_weights[i][0].requires_grad = False
#             self.gru_n.all_weights[i][1].requires_grad = False
#             self.gru_n.all_weights[i][2].requires_grad = False
#             self.gru_n.all_weights[i][3].requires_grad = False
#
#         self.n1.weight.requires_grad = False
#         self.n1.bias.requires_grad = False
#         self.n2.weight.requires_grad = False
#         self.n2.bias.requires_grad = False
#
#     def forward(self, x):
#
#         n, _ = self.gru_n(x.transpose(-1, -2))
#         n = self.n1(n)
#         n = n.transpose(-1, -2)
#         n = self.n2(n)
#
#         return n


class Model(nn.Module):

    def __init__(self, input_channels = 1, output_channels = 1, num_channels = [16, 16, 16, 32], L = 1200, wavelet_len=100):

        super(Model, self).__init__()

        self.encoder = Encoder(input_channels = input_channels, num_channels = num_channels, kernel_size = 3, dropout = 0.00)
        self.inverter = Inverter(input_channels = num_channels[-1], output_channels = output_channels)
        self.wavelet_exactor = Wavelet_Exactor(input_channels = num_channels[-1], output_channels = output_channels, seis_len = L, wavelet_len=wavelet_len)
        # self.liner = Liner(input_channels = num_channels[-1], mode = mode, L = L)
        # self.reconstructor = Reconstructor(input_channels = num_channels[-1], output_channels = output_channels)
        self.GN = nn.GroupNorm(num_channels = num_channels[-1], num_groups = input_channels)

    def partly_freeze(self):

        self.encoder.freeze()
        self.wavelet_exactor.freeze()
        # self.reconstructor.freeze()

    def forward(self, x):

        feature = self.encoder(x)
        feature = self.GN(feature)

        impedance = self.inverter(feature)
        wavelet = self.wavelet_exactor(feature)

        return impedance, wavelet


