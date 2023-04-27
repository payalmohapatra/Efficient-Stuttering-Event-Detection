""" 
Author : Payal Mohapatra
Contact : PayalMohapatra2026@u.northwestern.edu
Project : Speech Disfluency Detection using SSL Methods

This model architecture follows a temporal compression.
"""
# Utilities
import sys
import time
import matplotlib.pyplot as plt
import IPython.display as ipd
import argparse

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


class DisfluentSiam_temporal_comp(nn.Module):
    
    def __init__(self, dim=1024, pred_dim=256): # We can retain the same size of dimensions for now
        """
        dim: feature dimension (default: 1024)
        pred_dim: hidden dimension of the predictor (default: 512)
        The simsiam uses ResNets frozen layers --> We use wav2vec frozen layers.
        """
        super(DisfluentSiam_temporal_comp, self).__init__()

        # create the encoder
        """Create your custom pipeline.
           wav2vec2.0 output(directly obtain from dataloader) --> 2DConv --> BN --> 2DConv --> BN --> Flatten --> FC1 --> FC2 --> FC3
        """
        in_features = 768

        ################### Temporal Compression ####################################################################
        ## Shape of input here is (batch_size, 768, 149)
        self.layer1_conv1D = nn.Sequential(
            torch.nn.Conv1d(in_channels = in_features, out_channels = 384, 
                            dilation=2 ,kernel_size=3, stride=1, padding='valid'), # input to conv layers are (N, Cin, Lin)
            # Shape of output is (batch_size, 384, 145)
            torch.nn.ReLU(), 
            torch.nn.MaxPool2d(kernel_size=2, stride=2),
            #Shape of output is (batch_size, 192, 72)
            torch.nn.Dropout(p=0.5)
        )
        ## Do batch normalisation along each feature
        self.layer1_bn = nn.BatchNorm1d(192) # out_features/2 due to the maxpool layer
        ## Shape of output from here is (batch_size, 192, 72)

        self.layer2_conv1D = nn.Sequential(
            torch.nn.Conv1d(in_channels = 192, out_channels = 96, 
                            dilation=2 ,kernel_size=3, stride=1, padding='valid'), # input to conv layers are (N, Cin, Lin)
            #Shape of output is (batch_size, 96, 68)
            torch.nn.ReLU(),
            torch.nn.Dropout(p=0.5)
        )
        # Shape of output is (batch_size, 96 , 68)
        self.layer2_bn = nn.BatchNorm1d(96)

        # input size = (batch_size, 96, 62)
        self.flatten = torch.nn.Flatten()
        self.fc1 = nn.Sequential(nn.Linear(96*68,dim, bias=True),
                                 nn.BatchNorm1d(dim),
                                 nn.ReLU(inplace=True))
        self.fc2 = nn.Sequential(nn.Linear(dim,dim, bias=True),
                                 nn.BatchNorm1d(dim),
                                 nn.ReLU(inplace=True))
        # self.fc3 = nn.Sequential(nn.Linear(dim,dim, bias=True),
        #                          nn.BatchNorm1d(dim))
        
        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer

    def forward(self, x1, x2):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
        """
        ### Dont use numpy
        ## Reshape and permute before giving to 1D conv layer
        ## input shape is (batch_size, feature_size, sequence_length)
        batch_sz =   (x1).size()[0]
        feature_sz = (x1).size()[-1]
        seq_sz =     (x1).size()[-2]
        x1 = torch.reshape(x1, (batch_sz, seq_sz, feature_sz))
        x1 = x1.permute(0,2,1)
        x2 = torch.reshape(x2, (batch_sz, seq_sz, feature_sz))
        x2 = x2.permute(0,2,1)

        ## Call the first branch :: x1
        z1 = self.layer1_conv1D(x1)
        z1 = self.layer1_bn(z1)
        z1 = self.layer2_conv1D(z1)
        z1 = self.layer2_bn(z1)
        z1 = self.flatten(z1)
        z1 = self.fc1(z1)
        z1 = self.fc2(z1)

        ## Call the second branch :: x2
        z2 = self.layer1_conv1D(x2)
        z2 = self.layer1_bn(z2)
        z2 = self.layer2_conv1D(z2)
        z2 = self.layer2_bn(z2)
        z2 = self.flatten(z2)
        z2 = self.fc1(z2)
        z2 = self.fc2(z2)

        p1 = self.predictor(z1) # NxC
        p2 = self.predictor(z2) # NxC

        return p1, p2, z1.detach(), z2.detach()