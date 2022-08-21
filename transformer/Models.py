import torch
import torch.nn as nn

import transformer.Constants as Constants
from transformer.Layers import FFTBlock
from text.symbols import symbols
import hparams as hp

## Typing : v
## numpy  : v
## ModList: v

from typing import List

#@torch.jit.script
def cal_angle(position:int, hid_idx:int, d_hid:int) -> float:
    return position / (10000 ** (2 * (hid_idx // 2) / d_hid))
#@torch.jit.script
def get_posi_angle_vec(position:int, d_hid:int) -> torch.Tensor:
    return torch.tensor([cal_angle(position, hid_j, d_hid) for hid_j in range(d_hid)], dtype=torch.float)

#@torch.jit.script
def get_sinusoid_encoding_table(n_position:int,
                                      d_hid:int,
                                      padding_idx:int=-1) -> torch.Tensor:
    ''' Sinusoid position encoding table '''

    sinusoid_table:torch.Tensor = torch.stack([get_posi_angle_vec(pos_i, d_hid)
                                                    for pos_i in range(n_position)])

    sinusoid_table = sinusoid_table.transpose(0,1)

    s_table:List[torch.Tensor] = list()
    for i, s in enumerate(sinusoid_table):
        if i % 2 == 0:
            s_table.append(torch.sin(s))
        else:
            s_table.append(torch.cos(s))

    sinusoid_table = torch.stack(s_table)

    sinusoid_table = sinusoid_table.transpose(0,1)
            
    #s_table[:, 0::2] = torch.sin(sinusoid_table[:, 0::2])  # dim 2i
    #s_table[:, 1::2] = torch.cos(sinusoid_table[:, 1::2])  # dim 2i+1

    if padding_idx != -1:
        # zero vector for padding dimension
        sinusoid_table[padding_idx] = 0.
    
    return sinusoid_table


class Encoder(nn.Module):
    ''' Encoder '''
    #__constants__ = ['layer_stack']

    def __init__(self,
                 n_src_vocab : int =len(symbols)+1,
                 len_max_seq : int =hp.max_seq_len,
                 d_word_vec : int =hp.encoder_hidden,
                 n_layers : int =hp.encoder_layer,
                 n_head : int =hp.encoder_head,
                 d_k : int =hp.encoder_hidden // hp.encoder_head,
                 d_v : int =hp.encoder_hidden // hp.encoder_head,
                 d_model : int =hp.encoder_hidden,
                 d_inner : int =hp.fft_conv1d_filter_size,
                 dropout : float =hp.encoder_dropout):

        super(Encoder, self).__init__()

        n_position : int  = len_max_seq + 1

        self.src_word_emb = nn.Embedding(n_src_vocab, d_word_vec, padding_idx=Constants.PAD)
        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0), requires_grad=False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

    def forward(self,
                src_seq : torch.Tensor,
                mask : torch.Tensor,
                return_attns : bool =False) -> torch.Tensor:

        batch_size : int = src_seq.size(0)
        max_len : int = src_seq.size(1)
        
        # -- Prepare masks
        slf_attn_mask : torch.Tensor = mask.unsqueeze(1).expand(-1, max_len, -1)

        enc_output : torch.Tensor = torch.tensor([])
        # -- Forward
        if not self.training and src_seq.size(1) > hp.max_seq_len:
            enc_output = self.src_word_emb(src_seq) + get_sinusoid_encoding_table(src_seq.size(1), hp.encoder_hidden)[:src_seq.size(1), :].unsqueeze(0).expand(batch_size, -1, -1).to(src_seq.device)
        else:
            enc_output = self.src_word_emb(src_seq) + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        enc_slf_attn : torch.Tensor = torch.tensor([])
        for enc_layer in self.layer_stack:
            enc_output , enc_slf_attn = enc_layer(
                enc_output,
                mask=mask,
                slf_attn_mask=slf_attn_mask)

        return enc_output


class Decoder(nn.Module):
    """ Decoder """
    #__constants__ = ['layer_stack']
    
    def __init__(self,
                 len_max_seq=hp.max_seq_len,
                 d_word_vec=hp.encoder_hidden,
                 n_layers=hp.decoder_layer,
                 n_head=hp.decoder_head,
                 d_k=hp.decoder_hidden // hp.decoder_head,
                 d_v=hp.decoder_hidden // hp.decoder_head,
                 d_model=hp.decoder_hidden,
                 d_inner=hp.fft_conv1d_filter_size,
                 dropout=hp.decoder_dropout):

        super(Decoder, self).__init__()

        n_position : int = len_max_seq + 1

        self.position_enc = nn.Parameter(
            get_sinusoid_encoding_table(n_position, d_word_vec).unsqueeze(0), requires_grad=False)

        self.layer_stack = nn.ModuleList([FFTBlock(
            d_model, d_inner, n_head, d_k, d_v, dropout=dropout) for _ in range(n_layers)])

    def forward(self,
                enc_seq : torch.Tensor,
                mask : torch.Tensor,
                return_attns : bool =False) -> torch.Tensor:
        batch_size, max_len = enc_seq.size(0), enc_seq.size(1)

        # -- Prepare masks
        slf_attn_mask : torch.Tensor = mask.unsqueeze(1).expand(-1, max_len, -1)

        dec_output : torch.Tensor = torch.tensor([])
        # -- Forward
        if not self.training and enc_seq.size(1) > hp.max_seq_len:
            dec_output = enc_seq + get_sinusoid_encoding_table(enc_seq.size(1), hp.decoder_hidden)[:enc_seq.size(1), :].unsqueeze(0).expand(batch_size, -1, -1).to(enc_seq.device)
        else:
            dec_output = enc_seq + self.position_enc[:, :max_len, :].expand(batch_size, -1, -1)

        dec_slf_attn : torch.Tensor = torch.tensor([])
        for dec_layer in self.layer_stack:
            dec_output, dec_slf_attn = dec_layer(
                dec_output,
                mask,
                slf_attn_mask)

        return dec_output
