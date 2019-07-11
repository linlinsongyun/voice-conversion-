# -*- coding: utf-8 -*-
# !/usr/bin/env python

import tensorflow as tf
from tensorpack.graph_builder.model_desc import ModelDesc, InputDesc
from tensorpack.tfutils import (
    get_current_tower_context, optimizer, gradproc)
from tensorpack.tfutils.scope_utils import auto_reuse_variable_scope

import tensorpack_extension
#from data_load import phns
from hparam import hparam as hp
from modules import prenet, cbhg, normalize




class Net2(ModelDesc):

    def _get_inputs(self):
        #n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length + 1
        n_timesteps = (hp.default.duration * hp.default.sr) // hp.default.hop_length
        return [InputDesc(tf.float32, (None, n_timesteps, hp.default.n_mels), 'y_mel'), 
               InputDesc(tf.float32, (None, n_timesteps, hp.default.n_ppgs), 'ppgs'),]

    def _build_graph(self, inputs):
        #self.x_mfcc, self.y_spec, self.y_mel = inputs
        self.y_mel, self.ppgs = inputs
        is_training = get_current_tower_context().is_training

        # build net1
        '''
        self.net1 = Net1()
        with tf.variable_scope('net1'):
            self.ppgs, _, _ = self.net1.network(self.x_mfcc, is_training)
        self.ppgs = tf.identity(self.ppgs, name='ppgs')
        '''


        # build net2
        with tf.variable_scope('net2'):
            #self.pred_spec, self.pred_mel = self.network(self.ppgs, is_training)
            print("begin prediction")
            self.pred_mel = self.network(self.ppgs, is_training)
        #self.pred_spec = tf.identity(self.pred_spec, name='pred_spec')
        self.pred_mel = tf.identity(self.pred_mel, name = 'pred_mel')

        self.cost = self.loss()

        # summaries
        tf.summary.scalar('net2/train/loss', self.cost)

        if not is_training:
            tf.summary.scalar('net2/eval/summ_loss', self.cost)

    def _get_optimizer(self):
        gradprocs = [
            tensorpack_extension.FilterGradientVariables('.*net2.*', verbose=False),
            gradproc.MapGradient(
                lambda grad: tf.clip_by_value(grad, hp.train2.clip_value_min, hp.train2.clip_value_max)),
            gradproc.GlobalNormClip(hp.train2.clip_norm),
            # gradproc.PrintGradient(),
            # gradproc.CheckGradient(),
        ]
        global_step = tf.Variable(0, name='global_step',trainable=False)
        #self.lr = self.learning_rate_decay(global_step, hp.train2.lr)
        #lr = learning_rate_decay(initial_lr = hp.train2.lr, global_step)
        lr = tf.get_variable('learning_rate', initializer=hp.train2.lr, trainable=False)
        opt = tf.train.AdamOptimizer(learning_rate=lr)
        return optimizer.apply_grad_processors(opt, gradprocs)
    





    @auto_reuse_variable_scope
    def network(self, ppgs, is_training):
        # Pre-net
        prenet_out = prenet(ppgs,
                            num_units=[hp.train2.hidden_units, hp.train2.hidden_units // 2],
                            dropout_rate=hp.train2.dropout_rate,
                            is_training=is_training)  # (N, T, E/2)

        # CBHG1: mel-scale
        pred_mel = cbhg(prenet_out, hp.train2.num_banks, hp.train2.hidden_units // 2,
                        hp.train2.num_highway_blocks, hp.train2.norm_type, is_training,
                        scope="cbhg_mel")
        pred_mel = tf.layers.dense(pred_mel, self.y_mel.shape[-1], name='pred_mel')  # (N, T, n_mels)

        # CBHG2: linear-scale
        '''
        pred_spec = tf.layers.dense(pred_mel, hp.train2.hidden_units // 2)  # (N, T, n_mels)
        pred_spec = cbhg(pred_spec, hp.train2.num_banks, hp.train2.hidden_units // 2,
                   hp.train2.num_highway_blocks, hp.train2.norm_type, is_training, scope="cbhg_linear")
        pred_spec = tf.layers.dense(pred_spec, self.y_spec.shape[-1], name='pred_spec')  # log magnitude: (N, T, 1+n_fft//2)
        '''
        return pred_mel

    def loss(self):
        #loss_spec = tf.reduce_mean(tf.squared_difference(self.pred_spec, self.y_spec))
        #loss = tf.reduce_mean(tf.squared_difference(self.pred_mel, self.y_mel))
        loss = tf.reduce_mean(tf.abs(self.pred_mel-self.y_mel))
        #loss = loss_spec + loss_mel
        return loss

