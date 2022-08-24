import torch
import torch.nn as nn
import numpy as np
import os
import io

import hparams as hp
from fastspeech2 import FastSpeech2
from vocoder.vocgan_generator import Generator

device = torch.device('cpu')

def extract_fs():
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(350000))
    model = nn.DataParallel(torch.jit.script(FastSpeech2()))
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['model'])
    model.requires_grad = False
    model = torch.jit.optimize_for_inference(model.module.to(device))
    model.save("fastspeech.pt")

def extract_vg():
    n_mel_channels=hp.n_mel_channels
    generator_ratio = [4, 4, 2, 2, 2, 2]
    n_residual_layers=4
    mult=256
    out_channels=1

    checkpoint = torch.load(hp.vocoder_pretrained_model_path, map_location=device)
    model = Generator(n_mel_channels, n_residual_layers,
                        ratios=generator_ratio, mult=mult,
                        out_band=out_channels)

    model.load_state_dict(checkpoint['model_g'])
    model.requires_grad = False
    model = model.to(device)
    model.eval(True)
    model = torch.jit.optimize_for_inference(torch.jit.script(model))
    model.save("vocgan.pt")

def save_tensor(tensor, path):
    f = io.BytesIO()
    torch.save(tensor, f, _use_new_zipfile_serialization=True)
    with open(path, "wb") as out_f:
        # Copy the BytesIO stream to the output file
        out_f.write(f.getbuffer())

preprocessed_path = os.path.join("./preprocessed/", "kss", "mel_stat.npy")

if __name__ == "__main__":
    extract_fs()
    extract_vg()
    mean_mel, std_mel = torch.tensor(np.load(preprocessed_path), dtype=torch.float)

    save_tensor(mean_mel, "mean.pt")
    save_tensor(std_mel,  "std.pt")
