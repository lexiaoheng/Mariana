# -*- coding: utf-8 -*-
"""
Created on Wed Apr 10 20:40:49 2024

@author: Administrator
"""

import torch
from torch.nn.utils import weight_norm
from torch.nn.functional import conv1d
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm


# 剪切方法
class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, self.chomp_size:].contiguous()


# 注意力机制
class ScaledDotProductAttention(nn.Module):
    def __init__(self, d_k, dropout=0.2):
        super(ScaledDotProductAttention, self).__init__()
        self.d_k = d_k
        self.dropout = nn.Dropout(dropout)

    def forward(self, queries, keys, values):
        # queries, keys, values: (batch_size, num_heads, d_k, seq_len)
        scores = torch.matmul(queries, keys.transpose(-2, -1)) / (self.d_k ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.dropout(attention_weights)
        context = torch.matmul(attention_weights, values)
        return context, attention_weights


class AttentionBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, num_heads, d_k, dropout=0.2):
        super(AttentionBlock, self).__init__()
        self.q_linear = nn.Linear(n_inputs, n_outputs)
        self.k_linear = nn.Linear(n_inputs, n_outputs)
        self.v_linear = nn.Linear(n_inputs, n_outputs)
        self.attention = ScaledDotProductAttention(d_k, dropout)

    def forward(self, x):
        batch_size, _, seq_len = x.size()
        queries = self.q_linear(x.permute(0, 2, 1))
        keys = self.k_linear(x.permute(0, 2, 1))
        values = self.v_linear(x.permute(0, 2, 1))
        context, _ = self.attention(queries, keys, values)
        return context.permute(0, 2, 1)


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, num_heads, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = nn.Conv1d(n_inputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=True)
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = nn.Conv1d(n_outputs, n_outputs, kernel_size,
                               stride=stride, padding=padding, dilation=dilation, bias=True)
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.attention = AttentionBlock(n_outputs, n_outputs, num_heads, dropout)

        self.net = nn.Sequential(self.conv1, self.relu1, self.dropout1,
                                 self.conv2, self.relu2, self.dropout2,
                                 self.attention)

        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1, bias=True) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class GRUAttention(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, batch_first=True, bidirectional=True):
        super(GRUAttention, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bidirectional = bidirectional

        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=batch_first, bidirectional=bidirectional)
        self.attention = nn.Linear(hidden_size * (2 if bidirectional else 1), 1)

    def forward(self, input_seq):
        # 使用 GRU 编码输入序列
        output, hidden = self.gru(input_seq)

        # 根据 GRU 最终状态计算注意力权重
        attention_weights = F.softmax(self.attention(output.reshape(-1, output.size(-1))), dim=1)
        attention_weights = attention_weights.reshape(output.size(0), output.size(1), 1)

        # 计算加权的上下文向量
        context = torch.sum(output * attention_weights, dim=1)

        # 返回 output 和 hidden
        return output, hidden


# 反演网络
import torch
import torch.nn as nn
from torch.nn.utils import weight_norm
from torch.nn.functional import conv1d

class inverse_model(nn.Module):
    def __init__(self, resolution_ratio=1, nonlinearity="tanh"):
        super(inverse_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()

        self.cnn1 = nn.Sequential(TemporalBlock(n_inputs=1,
                                                n_outputs=16,
                                                kernel_size=5,
                                                stride=1,
                                                dilation=1,
                                                padding=2,

                                                num_heads=4,  # 设置注意力头的数量
                                                dropout=0.2),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=16))
        self.cnn2 = nn.Sequential(TemporalBlock(n_inputs=1,
                                                n_outputs=16,
                                                kernel_size=5,
                                                stride=1,
                                                dilation=3,
                                                padding=6,
                                                num_heads=4,  # 设置注意力头的数量
                                                dropout=0.2),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=16))
        self.cnn3 = nn.Sequential(TemporalBlock(n_inputs=1,
                                                n_outputs=16,
                                                kernel_size=5,
                                                stride=1,
                                                dilation=6,
                                                padding=12,
                                                num_heads=4,  # 设置注意力头的数量
                                                dropout=0.2),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=16))
        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=48,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=32),
                                 self.activation,

                                 nn.Conv1d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=32),
                                 self.activation,

                                 nn.Conv1d(in_channels=32,
                                           out_channels=32,
                                           kernel_size=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=32),
                                 self.activation)

        self.pointernet = GRUAttention(input_size=1, hidden_size=16, num_layers=3, batch_first=True, bidirectional=True)

        self.up = nn.Sequential(nn.ConvTranspose1d(in_channels=32,
                                                   out_channels=16,
                                                   stride=1,
                                                   kernel_size=3,
                                                   padding=1),
                                nn.GroupNorm(num_groups=1,
                                             num_channels=16),
                                self.activation,

                                nn.ConvTranspose1d(in_channels=16,
                                                   out_channels=16,
                                                   stride=1,
                                                   kernel_size=3,
                                                   padding=1),
                                nn.GroupNorm(num_groups=1,
                                             num_channels=16),
                                self.activation)

        self.pointernet_out = nn.GRU(input_size=16,
                                     hidden_size=16,
                                     num_layers=1,
                                     batch_first=True,
                                     bidirectional=True)
        self.out = nn.Linear(in_features=32, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1))

        tmp_x = x.transpose(-1, -2)
        rnn_out, _ = self.pointernet(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)

        x = rnn_out + cnn_out
        x = self.up(x)

        tmp_x = x.transpose(-1, -2)
        x, _ = self.pointernet_out(tmp_x)

        x = self.out(x)
        x = x.transpose(-1, -2)
        return x


# 正演网络

# class forward_model(nn.Module):
#     def __init__(self,resolution_ratio=4,nonlinearity="tanh"):
#         super(forward_model, self).__init__()
#         self.resolution_ratio = resolution_ratio
#         self.activation =  nn.ReLU() if nonlinearity=="relu" else nn.Tanh()

#         self.cnn1 = nn.Sequential(TemporalBlock(n_inputs=1, 
#                                                 n_outputs=16, 
#                                                 kernel_size=5, 
#                                                 stride=1, 
#                                                 dilation=1, 
#                                                 padding=2,

#                                                 num_heads=4,  # 设置注意力头的数量
#                                                 dropout=0.2),
#                                   nn.GroupNorm(num_groups=1,
#                                                 num_channels=16))
#         self.cnn2 = nn.Sequential(TemporalBlock(n_inputs=1, 
#                                                 n_outputs=16, 
#                                                 kernel_size=5, 
#                                                 stride=1, 
#                                                 dilation=3, 
#                                                 padding=6,
#                                                 num_heads=4,  # 设置注意力头的数量
#                                                 dropout=0.2),
#                                   nn.GroupNorm(num_groups=1,
#                                                 num_channels=16))
#         self.cnn3 = nn.Sequential(TemporalBlock(n_inputs=1, 
#                                                 n_outputs=16, 
#                                                 kernel_size=5, 
#                                                 stride=1, 
#                                                 dilation=6, 
#                                                 padding=12,
#                                                 num_heads=4,  # 设置注意力头的数量
#                                                 dropout=0.2),
#                                   nn.GroupNorm(num_groups=1,
#                                                 num_channels=16))
#         self.cnn = nn.Sequential(self.activation,
#                                  nn.Conv1d(in_channels=48,
#                                            out_channels=32,
#                                            kernel_size=3,
#                                            padding=1),
#                                  nn.GroupNorm(num_groups=1,
#                                               num_channels=32),
#                                  self.activation,

#                                  nn.Conv1d(in_channels=32,
#                                            out_channels=32,
#                                            kernel_size=3,
#                                            padding=1),
#                                  nn.GroupNorm(num_groups=1,
#                                               num_channels=32),
#                                  self.activation,

#                                  nn.Conv1d(in_channels=32,
#                                            out_channels=32,
#                                            kernel_size=1),
#                                  nn.GroupNorm(num_groups=1,
#                                               num_channels=32),
#                                  self.activation)

#         self.pointernet = GRUAttention(input_size=1, hidden_size=16, num_layers=3, batch_first=True, bidirectional=True)
#         #池化层
#         self.up = nn.Sequential(nn.Conv1d(in_channels=32,
#                                                    out_channels=16,
#                                                    stride=2,
#                                                    kernel_size=4,
#                                                    padding=1),
#                                 nn.GroupNorm(num_groups=1,
#                                              num_channels=16),
#                                 self.activation,

#                                 nn.Conv1d(in_channels=16,
#                                                    out_channels=16,
#                                                    stride=2,
#                                                    kernel_size=4,
#                                                    padding=1),
#                                 nn.GroupNorm(num_groups=1,
#                                              num_channels=16),
#                                 self.activation)


#         self.pointernet_out = nn.GRU(input_size=16,
#                               hidden_size=16,
#                               num_layers=1,
#                               batch_first=True,
#                               bidirectional=True)
#         self.out = nn.Linear(in_features=32, out_features=1)

#         for m in self.modules():
#             if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
#                 nn.init.xavier_uniform_(m.weight.data)
#                 if m.bias is not None:
#                     m.bias.data.zero_()
#             elif isinstance(m, nn.GroupNorm):
#                 m.weight.data.fill_(1)
#                 m.bias.data.zero_()
#             elif isinstance(m, nn.Linear):
#                 if m.bias is not None:
#                     m.bias.data.zero_()


#     def forward(self, x):
#         cnn_out1 = self.cnn1(x)
#         cnn_out2 = self.cnn2(x)
#         cnn_out3 = self.cnn3(x)
#         cnn_out = self.cnn(torch.cat((cnn_out1,cnn_out2,cnn_out3),dim=1))

#         tmp_x = x.transpose(-1, -2)
#         rnn_out, _ = self.pointernet(tmp_x)
#         rnn_out = rnn_out.transpose(-1, -2)

#         x = rnn_out + cnn_out
#         x = self.up(x)

#         tmp_x = x.transpose(-1, -2)
#         x, _ = self.pointernet_out(tmp_x)

#         x = self.out(x)
#         x = x.transpose(-1,-2)
#         return x


class forward_model(nn.Module):
    def __init__(self, resolution_ratio=1, nonlinearity="tanh"):
        super(forward_model, self).__init__()
        self.resolution_ratio = resolution_ratio
        self.activation = nn.ReLU() if nonlinearity == "relu" else nn.Tanh()

        self.cnn1 = nn.Sequential(TemporalBlock(n_inputs=1,
                                                n_outputs=8,
                                                kernel_size=5,
                                                stride=1,
                                                dilation=1,
                                                padding=2,

                                                num_heads=4,  # 设置注意力头的数量
                                                dropout=0.2),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn2 = nn.Sequential(TemporalBlock(n_inputs=1,
                                                n_outputs=8,
                                                kernel_size=5,
                                                stride=1,
                                                dilation=3,
                                                padding=6,
                                                num_heads=4,  # 设置注意力头的数量
                                                dropout=0.2),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn3 = nn.Sequential(TemporalBlock(n_inputs=1,
                                                n_outputs=8,
                                                kernel_size=5,
                                                stride=1,
                                                dilation=6,
                                                padding=12,
                                                num_heads=4,  # 设置注意力头的数量
                                                dropout=0.2),
                                  nn.GroupNorm(num_groups=1,
                                               num_channels=8))

        self.cnn = nn.Sequential(self.activation,
                                 nn.Conv1d(in_channels=24,
                                           out_channels=16,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=3,
                                           padding=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation,

                                 nn.Conv1d(in_channels=16,
                                           out_channels=16,
                                           kernel_size=1),
                                 nn.GroupNorm(num_groups=1,
                                              num_channels=16),
                                 self.activation)

        self.gru = GRUAttention(input_size=1, hidden_size=8, num_layers=3, batch_first=True, bidirectional=True)

        self.up = nn.Sequential(nn.Conv1d(in_channels=16,
                                          out_channels=8,
                                          stride=1,
                                          kernel_size=3,
                                          padding=1),
                                nn.GroupNorm(num_groups=1,
                                             num_channels=8),
                                self.activation,

                                nn.Conv1d(in_channels=8,
                                          out_channels=8,
                                          stride=1,
                                          kernel_size=3,
                                          padding=1),
                                nn.GroupNorm(num_groups=1,
                                             num_channels=8),
                                self.activation)

        self.gru_out = nn.GRU(input_size=8,
                              hidden_size=8,
                              num_layers=1,
                              batch_first=True,
                              bidirectional=True)
        self.out = nn.Linear(in_features=16, out_features=1)

        for m in self.modules():
            if isinstance(m, nn.Conv1d) or isinstance(m, nn.ConvTranspose1d):
                nn.init.xavier_uniform_(m.weight.data)
                m.bias.data.zero_()
            elif isinstance(m, nn.GroupNorm):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

    def forward(self, x):
        cnn_out1 = self.cnn1(x)
        cnn_out2 = self.cnn2(x)
        cnn_out3 = self.cnn3(x)
        cnn_out = self.cnn(torch.cat((cnn_out1, cnn_out2, cnn_out3), dim=1))

        tmp_x = x.transpose(-1, -2)
        rnn_out, _ = self.gru(tmp_x)
        rnn_out = rnn_out.transpose(-1, -2)

        x = rnn_out + cnn_out
        x = self.up(x)

        tmp_x = x.transpose(-1, -2)
        x, _ = self.gru_out(tmp_x)

        x = self.out(x)
        x = x.transpose(-1, -2)
        return x



