import torch
import torch.nn as nn
from torch.nn import functional as F
from collections import OrderedDict

from transformer.SubLayers import MultiHeadAttention, PositionwiseFeedForward

from typing import Tuple

## Typing : v
## shape  : v
## ModList: v


class FFTBlock(torch.nn.Module):
    """FFT Block"""

    def __init__(self,
                 d_model : int,
                 d_inner : int,
                 n_head : int,
                 d_k : int,
                 d_v : int,
                 dropout : float =0.1):
        super(FFTBlock, self).__init__()
        self.slf_attn = MultiHeadAttention(
            n_head, d_model, d_k, d_v, dropout=dropout)
        self.pos_ffn = PositionwiseFeedForward(
            d_model, d_inner, dropout=dropout)
    def forward(self,
                enc_input : torch.Tensor,
                mask : torch.Tensor,
                slf_attn_mask : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        enc_output : torch.Tensor = torch.tensor([])
        enc_slf_attn : torch.Tensor = torch.tensor([])
        enc_output , enc_slf_attn = self.slf_attn(
            enc_input, enc_input, enc_input, slf_attn_mask)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        enc_output = self.pos_ffn(enc_output)
        enc_output = enc_output.masked_fill(mask.unsqueeze(-1), 0)

        return enc_output, enc_slf_attn


class ConvNorm(torch.nn.Module):
    def __init__(self,
                 in_channels : int,
                 out_channels : int,
                 kernel_size : int =1,
                 stride : int =1,
                 padding : int =-1,
                 dilation : int =1,
                 bias : bool=True,
                 w_init_gain : str ='linear'):
        super(ConvNorm, self).__init__()

        if padding == -1:
            assert(kernel_size % 2 == 1)
            padding = int(dilation * (kernel_size - 1) / 2)

        self.conv = torch.nn.Conv1d(in_channels,
                                    out_channels,
                                    kernel_size=kernel_size,
                                    stride=stride,
                                    padding=padding,
                                    dilation=dilation,
                                    bias=bias)

    def forward(self,
                signal : torch.Tensor) -> torch.Tensor:
        conv_signal : torch.Tensor = self.conv(signal)

        return conv_signal


class PostNet(nn.Module):
    """
    PostNet: Five 1-d convolution with 512 channels and kernel size 5
    """
    #__constants__ = ['convolutions']

    def __init__(self,
                 n_mel_channels : int =80,
                 postnet_embedding_dim : int =512,
                 postnet_kernel_size : int =5,
                 postnet_n_convolutions : int =5):

        super(PostNet, self).__init__()
        self.convolutions = nn.ModuleList()

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(n_mel_channels,
                         postnet_embedding_dim,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='tanh'),

                nn.BatchNorm1d(postnet_embedding_dim))
        )

        for i in range(1, postnet_n_convolutions - 1):
            self.convolutions.append(
                nn.Sequential(
                    ConvNorm(postnet_embedding_dim,
                             postnet_embedding_dim,
                             kernel_size=postnet_kernel_size,
                             stride=1,
                             padding=int((postnet_kernel_size - 1) / 2),
                             dilation=1,
                             w_init_gain='tanh'),

                    nn.BatchNorm1d(postnet_embedding_dim))
            )

        self.convolutions.append(
            nn.Sequential(
                ConvNorm(postnet_embedding_dim,
                         n_mel_channels,
                         kernel_size=postnet_kernel_size,
                         stride=1,
                         padding=int((postnet_kernel_size - 1) / 2),
                         dilation=1,
                         w_init_gain='linear'),

                nn.BatchNorm1d(n_mel_channels))
        )

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        x : torch.Tensor = x.contiguous().transpose(1, 2)

        for i, conv in enumerate(self.convolutions):
            if (i != len(self.convolutions) - 1):
                x = F.dropout(torch.tanh(conv(x)), 0.5, self.training)
            else:
                x = F.dropout(           conv(x) , 0.5, self.training)

        x = x.contiguous().transpose(1, 2)
        return x
