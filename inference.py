# -*- coding: utf-8 -*-
# /usr/bin/python2


from __future__ import print_function

import argparse

from models import Net2
import numpy as np
import datetime
import tensorflow as tf
from hparam import hparam as hp
#from data_load import Net2DataFlow
from tensorpack.predict.base import OfflinePredictor
from tensorpack.predict.config import PredictConfig
from tensorpack.tfutils.sessinit import SaverRestore
from tensorpack.tfutils.sessinit import ChainInit
from tensorpack.callbacks.base import Callback
import librosa
import os
import wave
import struct
import time
def get_eval_input_names():
    return ['y_mel', 'ppgs']

def get_eval_output_names():
    return ['pred_mel']


def queue_input(ppgs_file, ppgs_dir, mel_path):
    ppgs_name = os.path.join(ppgs_dir, ppgs_file)
    ppgs = np.load(ppgs_name)
    n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length

    mel_name = get_mel_name(ppgs_file, mel_path)
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
    ppgs = ppgs.reshape((1, ppgs.shape[0], ppgs.shape[1]))
    mel = mel.reshape((1, mel.shape[0], mel.shape[1]))
    print("mel_nor",mel.shape)
    print("ppgs_nor",ppgs.shape)
    return mel, ppgs
    def get_mel_name(ppgs_file, mel_path):
    ppgs_name = ppgs_file.split('.npy')[0]
    print("ppgs_name",ppgs_name)
    mel_name = os.path.join(mel_path,'%s.mel.npy' % ppgs_name)
    print("mel_name", mel_name)
    return mel_name


def Net2DataFlow(ppgs_name, mel_dir):
    print("get into dataflow")
    while(True):
        yield queue_input(ppgs_name, mel_dir)

def ckpt2mel(predictor, ppgs_dir, mel_dir, save_dir):
    print("get into ckpt")
    for fi in os.listdir(ppgs_dir):
        print("fi",fi)
        #ppgs_name = os.path.join(ppgs_dir, fi)

        mel, ppgs = queue_input(fi, ppgs_dir, mel_dir)
        pred_mel = predictor(mel, ppgs)
        #print("pred_mel",pred_mel.size())
        pred_mel = np.array(pred_mel)
        print("pred_mel",pred_mel.shape)
        length = pred_mel.shape[2]
        width =  pred_mel.shape[3]
        pred_mel = pred_mel.reshape((length, width))
        save_name = fi.split('.npy')[0]
        if hp.default.n_mels == 20:
            npy_dir = os.path.join(save_dir,'lpc20')
            if not os.path.exists(npy_dir):
                os.makedirs(npy_dir)
            npy_path = os.path.join(npy_dir, '%s_20.npy' %save_name)
            np.save(npy_path, pred_mel)
            print('saved',npy_dir)if hp.default.n_mels == 32:
            npy_dir = os.path.join(save_dir,'lpc32')
            if not os.path.exists(npy_dir):
                os.makedirs(npy_dir)
            npy_path = os.path.join(npy_dir, '%s_32.npy' %save_name)
            np.save(npy_path, pred_mel)
            print('saved',npy_dir)def do_convert(args, logdir2):
    # Load graph
    model = Net2()
    index = 0
    ppgs_dir = hp.convert.ppgs_path
    mel_dir = hp.convert.mel_path
    #for fi in os.listdir(ppgs_dir):
    #print("fi",fi)
    #ppgs_path = os.path.join(ppgs_dir, fi)
    #df = Net2DataFlow(hp.convert.mel_path, ppgs_path, hp.convert.batch_size)
    #print("finish df")
    ckpt2 = '{}/{}'.format(logdir2, args.ckpt) if args.ckpt else tf.train.latest_checkpoint(logdir2)
    print("ckpt2",ckpt2)
    session_inits = []
    if ckpt2:
        session_inits.append(SaverRestore(ckpt2))
    pred_conf = PredictConfig( model=model,
                     input_names=get_eval_input_names(),
                     output_names=get_eval_output_names(),
                     session_init=ChainInit(session_inits))
    predictor = OfflinePredictor(pred_conf)
    print("after predictor")
    #import pdb;pdb.set_trace()
    ckpt2mel(predictor, ppgs_dir, mel_dir, hp.convert.save_path)
    print("success")
    def get_arguments():
    parser = argparse.ArgumentParser()
    parser.add_argument('case2', type=str, help='experiment case name of train2')
    parser.add_argument('-ckpt', help='checkpoint to load model.')
    arguments = parser.parse_args()
    return arguments


if __name__ == '__main__':
    args = get_arguments()
    hp.set_hparam_yaml(args.case2)
    logdir_train2 = '{}/{}/train2'.format(hp.logdir_path, args.case2)

    print('case2: {},logdir2: {}'.format(args.case2, logdir_train2))

    s = datetime.datetime.now()

    do_convert(args, logdir2=logdir_train2)

    e = datetime.datetime.now()
    diff = e - s
    print("Done. elapsed time:{}s".format(diff.seconds))
