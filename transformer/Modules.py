import torch
import torch.nn as nn

from typing import Tuple

## Typing : v
## numpy  : v
## ModList: v

class ScaledDotProductAttention(nn.Module):
    ''' Scaled Dot-Product Attention '''

    def __init__(self, temperature: float):
        super().__init__()
        self.temperature : float = temperature
        self.softmax = nn.Softmax(dim=2)

    def forward(self,
                q : torch.Tensor,
                k : torch.Tensor,
                v : torch.Tensor,
                mask : torch.Tensor) ->Tuple[torch.Tensor, torch.Tensor]:
        neg_inf : float = -float("inf")

        attn : torch.Tensor = torch.bmm(q, k.transpose(1, 2))
        attn = attn / self.temperature
        attn = attn.masked_fill(mask, neg_inf)
        attn = self.softmax(attn)
        output : torch.Tensor = torch.bmm(attn, v)

        return output, attn
