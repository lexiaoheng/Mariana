import torch.nn as nn
import torch
import torch.nn.functional as F

gru_init_std = 0.01
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

        # self.init_weights()

    # def init_weights(self):
    #
    #     for name, param in self.gru_out1.named_parameters():
    #         if 'weight' in name:
    #             nn.init.normal_(param, 0, 0.1)
    #         elif 'bias' in name:
    #             nn.init.normal_(param, 0, 0.1)
    #
    #     self.out.weight.data.normal_(0, gru_init_std)

    def forward(self, x):

        x, _ = self.gru_out1(x.transpose(-1, -2))
        x = self.out(x)
        x = x.transpose(-1, -2)

        return x


class Model(nn.Module):

    def __init__(self, input_channels = 1, output_channels = 1, num_channels = [16, 16, 16, 32]):

        super(Model, self).__init__()

        self.encoder = Encoder(input_channels = input_channels, num_channels = num_channels, kernel_size = 3, dropout = 0.00)
        self.inverter = Inverter(input_channels = num_channels[-1], output_channels = output_channels)

        self.GN = nn.GroupNorm(num_channels = num_channels[-1], num_groups = 1)

    def partly_freeze(self):

        self.encoder.freeze()
        self.wavelet_exactor.freeze()

    def forward(self, x):

        feature = self.encoder(x)
        feature = self.GN(feature)

        impedance = self.inverter(feature)

        return impedance


class SE(nn.Module):
    def __init__(self, input_size: int, seq_len:int, lstm_layer_num: int, hidden_size: int):

        super(SE, self).__init__()

        self.input_size = input_size
        self.seq_len = seq_len
        self.lstm = nn.GRU(input_size=self.input_size, hidden_size=hidden_size, bidirectional=True, batch_first=True,
                                 num_layers = lstm_layer_num)

        # self.linear = nn.Linear(hidden_size*2, seq_len)
        self.linear = nn.Linear(hidden_size * 2, 1)
        # self.Tanh = nn.Tanh()

    def forward(self, x):

        x1, _ = self.lstm(x)
        # x1 = self.Tanh(x1)
        x2 = self.linear(x1)

        return x1, x2
