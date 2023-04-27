""" 
Author : Payal Mohapatra
Contact : PayalMohapatra2026@u.northwestern.edu
Project : Speech Disfluency Detection using SSL Methods
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchaudio
import sys

import matplotlib.pyplot as plt
import IPython.display as ipd

from tqdm import tqdm

import os
import torch
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import os
import glob
import sklearn
from torch.utils.tensorboard import SummaryWriter
import random


##################################################################################################
# First things first! Set a seed for reproducibility.
# https://www.cs.mcgill.ca/~ksinha4/practices_for_reproducibility/
def set_seed(seed):
    """Set seed"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)
##################################################################################################
def count_model_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
##################################################################################################
"""All the unlabeled data are stored in a separate data folder. Assumes the follwoing hierarchy,
source_data_path 
    folder_1
        data1
        data2
        .
        .
        .
    folder_2
    folder_3
    .
    .
    .
This returns a list of all the file_paths.
"""
def _get_all_file_paths(source_data_path) :
        folder_list = os.listdir(source_data_path)
        num_folders  = len(folder_list)
        file_names = []
        for i in range(0, num_folders) :
            for file in os.listdir(os.path.join(source_data_path, folder_list[i])):
               d = os.path.join(source_data_path,folder_list[i], file)
               file_names.append(d)
        return file_names  


