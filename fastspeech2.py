import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer.Models import Encoder, Decoder
from transformer.Layers import PostNet
from modules import VarianceAdaptor
from utils import get_mask_from_lengths
import hparams as hp

from typing import Tuple, Optional

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class FastSpeech2(nn.Module):
    """ FastSpeech2 """

    def __init__(self, use_postnet=True):
        super(FastSpeech2, self).__init__()

        self.encoder = Encoder()
        self.variance_adaptor = VarianceAdaptor()

        self.decoder = Decoder()
        self.mel_linear = nn.Linear(hp.decoder_hidden, hp.n_mel_channels)
        
        self.use_postnet = use_postnet
        if self.use_postnet:
            self.postnet = PostNet()

    def forward(self,
                src_seq : torch.Tensor,
                src_len : torch.Tensor,
                mel_len : Optional[torch.Tensor] =None,
                d_target : Optional[torch.Tensor] =None,
                p_target : Optional[torch.Tensor] =None,
                e_target : Optional[torch.Tensor] =None,
                max_src_len : Optional[int]=None,
                max_mel_len : Optional[int]=None,
                dur_pitch_energy_aug : Optional[torch.Tensor] =None,
                f0_stat : Optional[torch.Tensor] =None,
                energy_stat : Optional[torch.Tensor] =None) :#-> Tuple[torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,torch.Tensor,Optional[torch.Tensor],Optional[torch.Tensor]]:
        src_seq = src_seq.to(torch.device('cpu')).long()
        src_len = src_len.to(torch.device('cpu')).long()

        src_mask = get_mask_from_lengths(src_len, max_src_len)
        mel_mask = get_mask_from_lengths(mel_len, max_mel_len) if mel_len is not None else None
        
        encoder_output = self.encoder(src_seq, src_mask)
        if d_target is not None:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, _, _ = self.variance_adaptor(
                encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len)
        else:
            variance_adaptor_output, d_prediction, p_prediction, e_prediction, mel_len, mel_mask = self.variance_adaptor(
                    encoder_output, src_mask, mel_mask, d_target, p_target, e_target, max_mel_len, dur_pitch_energy_aug, f0_stat, energy_stat)

        _mel_mask = mel_mask if mel_mask is not None else torch.tensor([False])
        decoder_output = self.decoder(variance_adaptor_output, _mel_mask)
        mel_output = self.mel_linear(decoder_output)
        
        if self.use_postnet:
            mel_output_postnet = self.postnet(mel_output) + mel_output
        else:
            mel_output_postnet = mel_output

        return mel_output_postnet
        #return mel_output, mel_output_postnet, d_prediction, p_prediction, e_prediction, src_mask, mel_mask, mel_len


if __name__ == "__main__":
    # Test
    model = FastSpeech2(use_postnet=False)
    print(model)
    print(sum(param.numel() for param in model.parameters()))
