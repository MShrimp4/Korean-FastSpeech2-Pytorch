#!/bin/python

from string import punctuation

import re

from g2pk import G2p
from jamo import h2j

import numpy as np
import torch

device='cpu'

PUNC = '!\'(),-.:;?'
SPACE = ' '

JAMO_LEADS = [chr(_) for _ in range(0x1100, 0x1113)]
JAMO_VOWELS = [chr(_) for _ in range(0x1161, 0x1176)]
JAMO_TAILS = [chr(_) for _ in range(0x11A8, 0x11C3)]

symbols = JAMO_LEADS + JAMO_VOWELS + JAMO_TAILS + list(PUNC)

##symbol_to_id = {s: i+2 for i, s in enumerate(symbols)}

def symbol_to_id (char):
    cint = ord(char)
    if      0x1100 <= cint < 0x1113:
        return cint - 0x1100 + 2
    elif 0x1161 <= cint < 0x1176:
        return cint - 0x1161 + (0x1113 - 0x1100) + 2
    elif 0x11A8 <= cint < 0x11C3:
        return cint - 0x11A8 + (0x1113 - 0x1100) + (0x1176 - 0x1161) + 2
    elif char in PUNC:
        return PUNC.find(char) + (0x1113 - 0x1100) + (0x1176 - 0x1161) + (0x11C3 - 0x11A8) + 2
    else:
        return 0

def list_to_sequence(lst):
    sequence = [symbol_to_id(e) for e in lst if e != ' ']
    return sequence

def kor_preprocess (text):
    text = text.rstrip(punctuation)
    
    g2p=G2p()
    phone = g2p(text)
    print('after g2p: ',phone)
    phone = h2j(phone)
    print('after h2j: ',phone)
    sequence = list_to_sequence(phone)
    sequence = np.array(sequence)
    sequence = np.stack([sequence])
    return sequence
