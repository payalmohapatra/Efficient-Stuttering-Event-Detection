"""
Author : Payal Mohapatra
Contact : PayalMohapatra2026@u.northwestern.edu
Project : Speech Disfluency Detection using SSL Methods
"""

# Utilities
import string
import sys
import time
import matplotlib.pyplot as plt
import IPython.display as ipd
import argparse
import math
from tqdm import tqdm 

#from __future__ import print_function, division
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

## Import custom functions from helper_functions file
from helper_functions import _get_all_file_paths
from helper_functions import set_seed

## Import audio augmentation functions
from audio_data_augment import _pitch_add_reverb
from audio_data_augment import _pitch_reverb

## Import the model
# import DisfluentSiam
import DisfluentSiam_temporal_comp

##################################################################################################
# Add arguments you want to pass via commandline
##################################################################################################
parser = argparse.ArgumentParser(description='SSL Speech Disfluency Training :: All disfluent data')
parser.add_argument('--batch_size', default=7, type=int,
                    metavar='N',
                    help='mini-batch size (default: 259), this is the total \n'
                         'batch size of all GPUs on the current node when \n'
                         'using Data Parallel or Distributed Data Parallel \n ' 
                         'Keep Batch size as multiples of GPU for predictable splits \n')
parser.add_argument('--init_lr', default=30, type=float,
                    metavar='N',
                    help='Initial learning rate with default set based on Adam optimiser\n.'
                    'If you change the lr decay schedule, then update the init_lr accordingly')
parser.add_argument('--num_epochs', default=1, type=int,
                    metavar='N',
                    help='Total number of training epochs')
parser.add_argument('--model_type', default='temp_compr', type=str,
                    metavar='N',
                    help='input which model you want to test \n'
                          'temp_compr = Uses the 1D CNN based model, ~10M parameters \n'
                          )    

parser.add_argument('--data_path', default='/home/payal/Efficient-Stuttering-Event-Detection/DataPrep/Sample_Data', type=str,
                    metavar='N',
                    help='The root data folder'
                          )                     
args = parser.parse_args()

##################################################################################################
# GPU related
##################################################################################################
set_seed(2711)
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")

##################################################################################################
# Dataloader for Audio Data
##################################################################################################
class AudioDataset(Dataset) :
    # Get a file list which you will traverse
    def __init__(self, path_list, n_samples) :
        # data loading
        self.path_list = path_list
        self.n_samples = n_samples

    def _get_audio_sample_path(self, path):
        signal, sr = torchaudio.load(path)
        signal = torchaudio.functional.resample(signal, orig_freq=sr, new_freq=16000) ## resample all the datasets to 16k Hz
        # print(np.shape(signal))
        ## Handle mismatched audio samples
        if (np.shape(signal)[1] !=48000) :
            print('Signals are not of the same length at :', path)
            if (np.shape(signal)[1] > 48000):
                signal = torch.reshape(signal, (1, 48000))
        
        # Call Augmentation 1 : Random Pitch Shift + Reverberation
        aug_1 = _pitch_reverb(signal, 16000)

        # Call Augmentation 2 : Random Pitch Shift + Noise + Reverberation 
        aug_2 = _pitch_add_reverb(signal, 16000)

        bundle = torchaudio.pipelines.WAV2VEC2_BASE
        model_wav2vec = bundle.get_model()
        aug_feat_1, _ = model_wav2vec(aug_1)
        aug_feat_2, _ = model_wav2vec(aug_2)

        aug_feat_1    = aug_feat_1.cpu().detach().numpy()        
        aug_feat_2    = aug_feat_2.cpu().detach().numpy()
        return aug_feat_1, aug_feat_2
  
           
    def __getitem__(self,index) :
        # path  = self._get_all_file_paths(self.folder_path)
        return self._get_audio_sample_path(self.path_list[index])
    def __len__(self) :    
        return self.n_samples   


##################################################################################################
""" Update this to pass necessary arguments via commandline
"""
lr = args.init_lr
batch_size = args.batch_size ## Multiples of 7
num_epochs = args.num_epochs
# create model
print("=> creating model")

model = DisfluentSiam_temporal_comp.DisfluentSiam_temporal_comp().to(device)


print(model)

init_lr = lr * batch_size / 259 ## TODO :: Can I have a better schedule learning rate?

############# Get all the paths from all unlabeled dataset #####################################

root_data_path = args.data_path
## Get all paths from fbank
source_data_path_fbank = root_data_path + '/Fluency_Bank'
path_list_fbank = _get_all_file_paths(source_data_path_fbank)

## Get all paths from UCLASS
def absoluteFilePaths(directory): ## helper function for absolute paths
    file_names = []
    for dirpath,_,filenames in os.walk(directory):
        for f in filenames:
            d =  os.path.abspath(os.path.join(dirpath, f))
            file_names.append(d)
        return file_names   

path_list_uclass_r1 = absoluteFilePaths(root_data_path + '/UCLASS_R1')
path_list_uclass_r2 = absoluteFilePaths(root_data_path + '/UCLASS_R2')

## Get all paths from TORGO
path_list_torgo = _get_all_file_paths(root_data_path + '/TORGO')
path_list = path_list_fbank + path_list_uclass_r1 + path_list_uclass_r2 + path_list_torgo


print("Training DisfluentSiam on ", (len(path_list)*3/3600), 'hours of data.')

train_dataset = AudioDataset(path_list, len(path_list))

train_loader = DataLoader(dataset=train_dataset,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=1, drop_last=True, pin_memory=True)


criterion = nn.CosineSimilarity()
optimizer = torch.optim.Adam(model.parameters(), init_lr, weight_decay=1e-4)

##################################################################################################
writer = SummaryWriter()
writer = SummaryWriter('SSL Speech Disfluency Training :: SEP-28k Adam warmup lr')
writer = SummaryWriter(comment='SSL Speech Disfluency Training :: SEP-28k adam warmup decaying lr')
##################################################################################################
class AverageMeter():
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        return self.avg

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter():
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'

        
def train_one_epoch(train_loader, model, criterion, optimizer, epoch):
    loss_list = np.zeros(len(train_loader)) 
    
    batch_time = AverageMeter('Batch Time', ':6.3f')
    data_time = AverageMeter('Data Time', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader), # total_data/batch_size
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    # for i, (image1, image2) in tqdm(enumerate(train_loader), total=len(train_loader), leave=False):
    for i, (image1, image2) in enumerate(train_loader):
        image1 = image1.to(device)
        image2 = image2.to(device)
        
        
        # measure data loading time
        data_time.update(time.time() - end)

        # compute output and loss
        p1, p2, z1, z2 = model(image1, image2)
        
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5 # @Payal :: the forward method returns a detached z1 and z2. Note the criterion is negated so that we maximise the cosine similarity.

        curr_loss = losses.update(loss.item(), image1.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        loss_list[i] = curr_loss
        if i % 2 == 0:
            progress.display(i)
    return loss_list 

# train_one_epoch(train_loader, model, criterion, optimizer, 1)
def adjust_learning_rate_cosine_anealing(optimizer, init_lr, epoch, num_epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / num_epochs))
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    print('Learning rate inside adjusting cosine lr = ', cur_lr)

def adjust_learning_rate_warmup_time(optimizer, init_lr, epoch, num_epochs, model_size, warmup):
    """Decay the learning rate based on warmup schedule based on time
    Source :: https://nlp.seas.harvard.edu/2018/04/03/attention.html#optimizer 
    """
    cur_lr = (model_size ** (-0.5) * min((epoch+1) ** (-0.5), (epoch+1) * warmup ** (-1.5))) 
    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    print('Learning rate inside adjusting warmup decay lr = ', cur_lr)

def naive_lr_decay(optimizer, init_lr, epoch, num_epochs):
    """
    Make 3 splits in the num_epochs and just use that to decay the lr 
    """
    if (epoch < np.ceil(num_epochs/4)) :
        cur_lr = init_lr
    elif (epoch < np.ceil(num_epochs/2)) :
        cur_lr = 0.5 * init_lr
    else :
        cur_lr = 0.25 * init_lr    

    for param_group in optimizer.param_groups:
        param_group['lr'] = cur_lr
    print('Learning rate inside naive decay lr = ', cur_lr)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
model_train_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

loss_list_epoch = np.zeros(num_epochs)
for epoch in range(0, num_epochs):
        naive_lr_decay(optimizer, init_lr, epoch, num_epochs)

        # train for one epoch
        loss_epoch = train_one_epoch(train_loader, model, criterion, optimizer, epoch)
        # visualisation per epoch
        # average loss through all iterations
        writer.add_scalar("Loss/train", sum(loss_epoch)/len(loss_epoch), epoch) 

        # if not args.multiprocessing_distributed or (args.multiprocessing_distributed
        #         and args.rank % ngpus_per_node == 0):
        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'optimizer' : optimizer.state_dict(),
        }, filename='dummy_path_{:04d}.pth.tar'.format(epoch))

        loss_list_epoch[epoch] = np.mean(loss_epoch)


writer.close()

## Save Model
# torch.save(model, 'Model_Loss_Logs/Aug_31_all_data_run//Aug_31_loss.pth')


# Save the loss
# df_loss = pd.DataFrame(loss_list_epoch)
# df_loss.to_csv('Model_Loss_Logs/Aug_31_all_data_run//Aug_31_loss.csv')


