import torch.nn as nn
import torch.nn.functional as F

from transformer.Modules import ScaledDotProductAttention
import hparams as hp

import torch
from typing import Tuple

## Typing : v
## numpy  : v
## ModList: v

class MultiHeadAttention(nn.Module):
    ''' Multi-Head Attention module '''

    def __init__(self,
                 n_head : int,
                 d_model : int,
                 d_k : int,
                 d_v : int,
                 dropout : float =0.1):
        super().__init__()

        self.n_head : int = n_head
        self.d_k : int = d_k
        self.d_v : int = d_v

        self.w_qs = nn.Linear(d_model, n_head * d_k)
        self.w_ks = nn.Linear(d_model, n_head * d_k)
        self.w_vs = nn.Linear(d_model, n_head * d_v)
        
        self.attention = ScaledDotProductAttention(
            temperature=d_k ** 0.5)
        self.layer_norm = nn.LayerNorm(d_model)

        self.fc = nn.Linear(n_head * d_v, d_model)

        self.dropout = nn.Dropout(dropout)

    def forward(self,
                q : torch.Tensor,
                k : torch.Tensor,
                v : torch.Tensor,
                mask : torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:

        d_k : int = self.d_k
        d_v : int = self.d_v
        n_head : int = self.n_head

        sz_b : int = q.size(0)
        len_q : int = q.size(1)
        sz_b : int = k.size(0)
        len_k : int = k.size(1)
        sz_b : int = v.size(0)
        len_v : int = v.size(1)

        residual : torch.Tensor = q

        q : torch.Tensor = self.w_qs(q).view(sz_b, len_q, n_head, d_k)
        k : torch.Tensor = self.w_ks(k).view(sz_b, len_k, n_head, d_k)
        v : torch.Tensor = self.w_vs(v).view(sz_b, len_v, n_head, d_v)
        q = q.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_q, d_k)  # (n*b) x lq x dk
        k = k.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_k, d_k)  # (n*b) x lk x dk
        v = v.permute(2, 0, 1, 3).contiguous().view(-1,
                                                    len_v, d_v)  # (n*b) x lv x dv

        mask = mask.repeat(n_head, 1, 1)  # (n*b) x .. x ..

        output : torch.Tensor = torch.tensor([])
        attn : torch.Tensor = torch.tensor([])
        output, attn = self.attention(q, k, v, mask)

        output = output.view(n_head, sz_b, len_q, d_v)
        output = output.permute(1, 2, 0, 3).contiguous().view(
            sz_b, len_q, -1)  # b x lq x (n*dv)

        output = self.dropout(self.fc(output))
        output = self.layer_norm(output + residual)

        return output, attn


class PositionwiseFeedForward(nn.Module):
    ''' A two-feed-forward-layer module '''

    def __init__(self,
                 d_in : int,
                 d_hid : int,
                 dropout : float =0.1):
        super().__init__()

        # Use Conv1D
        # position-wise
        self.w_1 = nn.Conv1d(
            d_in, d_hid, kernel_size=hp.fft_conv1d_kernel_size[0], padding=(hp.fft_conv1d_kernel_size[0]-1)//2)
        # position-wise
        self.w_2 = nn.Conv1d(
            d_hid, d_in, kernel_size=hp.fft_conv1d_kernel_size[1], padding=(hp.fft_conv1d_kernel_size[1]-1)//2)

        self.layer_norm = nn.LayerNorm(d_in)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x : torch.Tensor) -> torch.Tensor:
        residual : torch.Tensor = x
        output : torch.Tensor = x.transpose(1, 2)
        output = self.w_2(F.relu(self.w_1(output)))
        output = output.transpose(1, 2)
        output = self.dropout(output)
        output = self.layer_norm(output + residual)

        return output
