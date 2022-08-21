import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict
import copy
import math

import hparams as hp
import utils

from typing import Optional, List, Tuple

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

## Typing : 
## numpy  : v
## ModList: 

#def clones(module, N):
#    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

class VarianceAdaptor(nn.Module):
    """ Variance Adaptor """

    def __init__(self):
        super(VarianceAdaptor, self).__init__()
        self.duration_predictor = VariancePredictor()
        self.length_regulator = LengthRegulator()
        self.pitch_predictor = VariancePredictor()
        self.energy_predictor = VariancePredictor()

        self.energy_embedding_producer = Conv(1, hp.encoder_hidden, kernel_size=9, bias=False, padding=4)    
        self.pitch_embedding_producer = Conv(1, hp.encoder_hidden, kernel_size=9, bias=False, padding=4)

    def forward(self,
                x : torch.Tensor,
                src_mask : torch.Tensor,
                mel_mask : Optional[torch.Tensor]=None,
                duration_target : Optional[torch.Tensor]=None,
                pitch_target : Optional[torch.Tensor]=None,
                energy_target : Optional[torch.Tensor]=None,
                max_len : Optional[int]=None,
                dur_pitch_energy_aug : Optional[torch.Tensor]=None,
                f0_stat : Optional[torch.Tensor]=None,
                energy_stat : Optional[torch.Tensor]=None) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, Optional[torch.Tensor]]:
        log_duration_prediction : torch.Tensor = self.duration_predictor(x, src_mask)
        dpe : torch.Tensor = dur_pitch_energy_aug if dur_pitch_energy_aug is not None else torch.tensor([1.0,1.0,1.0])
        
        pitch_prediction : torch.Tensor = self.pitch_predictor(x, src_mask)
        if pitch_target is not None:
            pitch_embedding = self.pitch_embedding_producer(pitch_target.unsqueeze(2))
        else:
            f0s : torch.Tensor = f0_stat if f0_stat is not None else torch.tensor([0,1]) ## mean = 0, std = 1
            pitch_prediction = utils.de_norm(pitch_prediction, mean=f0s[0], std=f0s[1]) * dpe[1]
            pitch_prediction = utils.standard_norm_torch(pitch_prediction, mean=f0s[0], std=f0s[1])
            pitch_embedding = self.pitch_embedding_producer(pitch_prediction.unsqueeze(2))
    
        energy_prediction = self.energy_predictor(x, src_mask)
        if energy_target is not None:
            energy_embedding = self.energy_embedding_producer(energy_target.unsqueeze(2))
        else:
            es : torch.Tensor = energy_stat if energy_stat is not None else torch.tensor([0,1]) ## mean = 0, std = 1
            energy_prediction = utils.de_norm(energy_prediction, mean=es[0], std=es[1]) * dpe[2]
            energy_prediction = utils.standard_norm_torch(energy_prediction, mean=es[0], std=es[1])
            energy_embedding = self.energy_embedding_producer(energy_prediction.unsqueeze(2)) 

        x = x + pitch_embedding + energy_embedding

        if duration_target is not None:
            x, mel_len = self.length_regulator(x, duration_target, max_len)
        else:
            duration_rounded = torch.clamp(torch.round(torch.exp(log_duration_prediction)-hp.log_offset) * dpe[0], min=0)
            x, mel_len = self.length_regulator(x, duration_rounded, max_len)
            mel_mask = utils.get_mask_from_lengths(mel_len)
        
        return x, log_duration_prediction, pitch_prediction, energy_prediction, mel_len, mel_mask


class LengthRegulator(nn.Module):
    """ Length Regulator """

    def __init__(self):
        super(LengthRegulator, self).__init__()

    def LR(self, x, duration, max_len : Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        output_list : List[torch.Tensor] = list()
        mel_len : List[int] = list()
        for batch, expand_target in zip(x, duration):
            expanded = self.expand(batch, expand_target)
            output_list.append(expanded)
            mel_len.append(expanded.size(0))

        if max_len is not None:
            output = utils.pad(output_list, max_len)
        else:
            output = utils.pad(output_list)

        return output, torch.tensor(mel_len, dtype=torch.long)

    def expand(self, batch, predicted):
        out : List[torch.Tensor] = list()

        for i, vec in enumerate(batch):
            expand_size = predicted[i].item()
            out.append(vec.expand(int(expand_size), -1))
        out_tensor = torch.cat(out, 0)

        return out_tensor

    def forward(self, x, duration, max_len : Optional[int]) -> Tuple[torch.Tensor, torch.Tensor]:
        output, mel_len = self.LR(x, duration, max_len)
        return output, mel_len


class VariancePredictor(nn.Module):
    """ Duration, Pitch and Energy Predictor """

    def __init__(self):
        super(VariancePredictor, self).__init__()

        self.input_size = hp.encoder_hidden
        self.filter_size = hp.variance_predictor_filter_size
        self.kernel = hp.variance_predictor_kernel_size
        self.conv_output_size = hp.variance_predictor_filter_size
        self.dropout = hp.variance_predictor_dropout

        self.conv_layer = nn.Sequential(OrderedDict([
            ("conv1d_1", Conv(self.input_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=(self.kernel-1)//2)),
            ("relu_1", nn.ReLU()),
            ("layer_norm_1", nn.LayerNorm(self.filter_size)),
            ("dropout_1", nn.Dropout(self.dropout)),
            ("conv1d_2", Conv(self.filter_size,
                              self.filter_size,
                              kernel_size=self.kernel,
                              padding=1)),
            ("relu_2", nn.ReLU()),
            ("layer_norm_2", nn.LayerNorm(self.filter_size)),
            ("dropout_2", nn.Dropout(self.dropout))
        ]))

        self.linear_layer = nn.Linear(self.conv_output_size, 1)

    def forward(self, encoder_output, mask : Optional[torch.Tensor]):
        out = self.conv_layer(encoder_output)
        out = self.linear_layer(out)
        out = out.squeeze(-1)
        
        if mask is not None:
            out = out.masked_fill(mask, 0.)
        
        return out


class Conv(nn.Module):
    """
    Convolution Module
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 kernel_size=1,
                 stride=1,
                 padding=0,
                 dilation=1,
                 bias=True,
                 w_init='linear'):
        """
        :param in_channels: dimension of input
        :param out_channels: dimension of output
        :param kernel_size: size of kernel
        :param stride: size of stride
        :param padding: size of padding
        :param dilation: dilation rate
        :param bias: boolean. if True, bias is included.
        :param w_init: str. weight inits with xavier initialization.
        """
        super(Conv, self).__init__()

        self.conv = nn.Conv1d(in_channels,
                              out_channels,
                              kernel_size=kernel_size,
                              stride=stride,
                              padding=padding,
                              dilation=dilation,
                              bias=bias)

    def forward(self, x):
        x = x.contiguous().transpose(1, 2)
        x = self.conv(x)
        x = x.contiguous().transpose(1, 2)

        return x
