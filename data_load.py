# -*- coding: utf-8 -*-

import glob
import random

import librosa
import numpy as np
from tensorpack.dataflow.base import RNGDataFlow
from tensorpack.dataflow.common import BatchData
from tensorpack.dataflow import PrefetchData
from hparam import hparam as hp
from utils import normalize_0_1
import os

class DataFlow(RNGDataFlow):

    def __init__(self, mel_path, ppgs_path, batch_size):
        self.batch_size = batch_size
        self.mel_path = mel_path
        self.ppgs_files = glob.glob(ppgs_path)
               
#        print("mel_path",mel_path)

    def __call__(self, n_prefetch=1000, n_thread=1):
        df = self
        df = BatchData(df, self.batch_size)
        df = PrefetchData(df, n_prefetch, n_thread)
        return df



class Net2DataFlow(DataFlow):
    #print("start")
    def get_data(self):
        while True:
            #wav_file = random.choice(self.wav_files)
            #print("self.mel_files",self.mel_files)
            #import pdb;pdb.set_trace()
            ppgs_file = random.choice(self.ppgs_files)
           # print("ppgs_file",ppgs_file)
            
            yield queue_input(ppgs_file,self.mel_path)
            #print("wav_file",wav_file)
            #yield get_mfccs_and_spectrogram(wav_file)
 
def queue_input(ppgs_file,mel_path):
    ppgs = np.load(ppgs_file)
    n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length 

    mel_name = get_mel_name(ppgs_file,mel_path)
    #print("mel_name",mel_name)
    mel = np.load(mel_name)
    a = mel.shape[0]-ppgs.shape[0]
    mel = mel[:-a]
    if mel.shape[0] < n_timesteps :
        x = n_timesteps - mel.shape[0]
        mel = np.pad(mel,((0,x),(0,0)),'constant', constant_values=(0,0))
        ppgs = np.pad(ppgs, ((0,x),(0,0)),'constant',constant_values=(0,0))
    else:
        mel = mel[:n_timesteps]
        ppgs = ppgs[:n_timesteps]
    #print("mel_nor",mel.shape)
    #print("ppgs_nor",ppgs.shape)
    return mel, ppgs


def get_mel_name(ppgs_file, mel_path):
    ppgs_name = ppgs_file.split('/')
    ppgs_name = ppgs_name[-1].split('.npy')
    mel_name = ppgs_name[0]
    mel_name = os.path.join(mel_path,'%s.mel.npy'%mel_name)
    return mel_name
