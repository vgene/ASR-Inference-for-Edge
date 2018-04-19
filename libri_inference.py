import scipy.io.wavfile as wav
from sklearn import preprocessing

import time
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

# from speechvalley.utils import count_params
from dynamic_brnn import DBiRNN
from utils import dotdict, describe, output_to_sequence
from calcmfcc import calcfeat_delta_delta

def getFeature(filename, mode = 'mfcc', feature_len =13, win_step = 0.01, win_len = 0.02)
	(rate,sig)= wav.read(filename)
	feat = calcfeat_delta_delta(sig,rate,
	    win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
	feat = preprocessing.scale(feat)
	feat = np.transpose(feat)
	return feat

def getResult(args, audio_file):
    feat = getFeature(audio_file)
    seqLength = feat.shape[1]
    maxTimeSteps = feat.shape[1]
    model = DBiRNN(args, maxTimeSteps)

    # num_params = count_params(model, mode='trainable')
    # all_num_params = count_params(model, mode='all')
    # model.config['trainable params'] = num_params
    # model.config['all params'] = all_num_params
    print(model.config)

    with tf.Session(graph=model.graph) as sess:
        # restore from stored model
        ckpt = tf.train.get_checkpoint_state(args.savedir)
        if ckpt and ckpt.model_checkpoint_path:
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored from:' + savedir)

	    batchInputs = [feat]
	    batchSeqLengths = [seqLength]
	    feedDict = {model.inputX: batchInputs, model.seqLengths: batchSeqLengths}

	    l, pre = sess.run([model.loss, model.predictions], feed_dict=feedDict)

	    batchErrors[batch] = er
	    print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},test loss={:.3f},mean test CER={:.3f}\n'.format(
	        level, totalN, id_dir+1, len(feature_dirs), batch+1, len(batchRandIxs), l, er/batch_size))

	    print('Output:\n' + output_to_sequence(pre, type=level))

def main():
	args['mode'] = 'test'
	args['level'] = 'cha'
	args['model'] = 'DBiRNN'
	args['rnncell'] = 'lstm'
	args['num_layer'] = 2
	args['activation'] = 'tanh'
	args['batch_size'] = 1
	args['num_hidden'] = 256
	args['num_feature'] = 39
	args['num_classes'] = 30
	args['num_epochs'] = 1
	args['savedir'] = '/home/zyxu/libri_test'
	args = dotdict(args)

	getResult(args, "/home/zyxu/libri_test/test.wav")
