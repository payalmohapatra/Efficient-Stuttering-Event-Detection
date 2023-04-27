""" Created on : 14th July
Author : Payal Mohapatra
Contact : PayalMohapatra2026@u.northwestern.edu
Project : Speech Disfluency Detection using SSL Methods
"""
# Utilities
import sys
import time
import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
import sklearn

import random


import augment
# import augment

## General pytorch libraries
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio

## Import audio related packages
import torchaudio
import torchaudio.functional as F
import torchaudio.transforms as T
import librosa as lr
from IPython.display import Audio
import soundfile as sf
from python_speech_features import mfcc, logfbank
import soundfile as sf


# Generate a random shift applied to the speaker's pitch
def random_pitch_shift():
    return np.random.randint(-300, 300)

# Generate a random size of the room
def random_room_size():
    return np.random.randint(0, 100)

noise_generator = lambda: torch.zeros_like(signal).uniform_()

def _pitch_reverb(audio_waveform, sr):
        combination = augment.EffectChain() \
            .pitch("-q", random_pitch_shift).rate(sr) \
            .reverb(50, 50, random_room_size).channels(1) 
        y = combination.apply(audio_waveform, src_info={'rate': sr}, target_info={'rate': sr})
        return y


def _pitch_add_reverb(audio_waveform, sr):
    noise_generator = lambda: torch.zeros_like(audio_waveform).uniform_()
    combination = augment.EffectChain() \
        .pitch("-q", random_pitch_shift).rate(sr) \
        .additive_noise(noise_generator, snr=5) \
        .reverb(50, 50, random_room_size).channels(1) 
    y = combination.apply(audio_waveform, src_info={'rate': sr}, target_info={'rate': sr})
    return y
