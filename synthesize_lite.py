import onnx
import onnxruntime

import torch
import torch.nn as nn
import numpy as np
import os

from scipy.io import wavfile


import hparams as hp ## TODO REMOVE
from fastspeech2 import FastSpeech2
from vocoder import vocgan_generator
from kor_preprocess import kor_preprocess

import audio as Audio


from vocoder import vocgan_generator #is this required?
import utils

from modules import LengthRegulator


device = 'cpu'#torch.device('cuda' if torch.cuda.is_available() else 'cpu')
fastONNX_path = os.path.join("./onnx", "FastSpeech2.onnx")

def vocgan_infer(mel, vocoder, path):
    model = vocoder

    with torch.no_grad():
        if len(mel.shape) == 2:
            mel = mel.unsqueeze(0)

        audio = model.infer(mel).squeeze()
        audio = hp.max_wav_value * audio[:-(hp.hop_length*10)]
        audio = audio.clamp(min=-hp.max_wav_value, max=hp.max_wav_value-1)
        audio = audio.short().cpu().detach().numpy()

        wavfile.write(path, hp.sampling_rate, audio)        

preprocessed_path = os.path.join("./preprocessed/", "kss", "mel_stat.npy")
test_path         = "./results"

def synthesize(model, vocoder, text, sentence, prefix=''):
    file_name = sentence[:10] # long filename will result in OS Error

    mean_mel, std_mel = torch.tensor(np.load(preprocessed_path), dtype=torch.float).to(device)

    mean_mel, std_mel = mean_mel.reshape(1, -1), std_mel.reshape(1, -1)

    src_len = np.array([text.shape[1]])

    text    = torch.from_numpy(text).long().to(device)
    src_len = torch.from_numpy(src_len).to(device)
    mel_postnet = model(text, src_len)
    mel_postnet_torch = utils.de_norm(mel_postnet, mean_mel, std_mel).transpose(1, 2)

    if not os.path.exists(test_path):
        os.makedirs(test_path)

    Audio.tools.inv_mel_spec(mel_postnet_torch[0], os.path.join(test_path, '{}_griffin_lim_{}.wav'.format(prefix, file_name)))

    utils.vocgan_infer(mel_postnet_torch, vocoder, path=os.path.join(test_path, '{}_{}_{}.wav'.format(prefix, hp.vocoder, file_name)))


#print("gen")
gen_FastSpeech2()
vocoder = utils.get_vocgan(ckpt_path=hp.vocoder_pretrained_model_path)
#synthesize(get_FastSpeech2(), vocoder, np.stack([np.random.randint(44, size=2)]), "종마리", prefix='step_{}'.format(350000))
synthesize(get_FastSpeech2(), vocoder, kor_preprocess("인간 시대의 종말이 도래했다"), "종마리", prefix='step_{}'.format(350000))
#print("get")
#get_ONNX_FastSpeech2()
#print("synth")
#onnx.save(onnx.shape_inference.infer_shapes(onnx.load(fastONNX_path)), fastONNX_path)
synthesize(get_ONNX_FastSpeech2(), vocoder, kor_preprocess("오늘부터 이 도시는 배틀시티"), "종마리_", prefix='step_onnx_{}'.format(350000))
    
