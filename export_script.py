import torch
import torch.nn as nn
import numpy as np
import os

import hparams as hp
from fastspeech2 import FastSpeech2

def extract():
    checkpoint_path = os.path.join(hp.checkpoint_path, "checkpoint_{}.pth.tar".format(350000))
    model = nn.DataParallel(torch.jit.script(FastSpeech2()))
    model.load_state_dict(torch.load(checkpoint_path, map_location=torch.device('cpu'))['model'])
    model.requires_grad = False
    model = model.module.to(torch.device('cpu'))
    model.save("fastspeech.pt")

extract()
