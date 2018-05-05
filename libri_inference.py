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

activation_functions_dict = {
    'sigmoid': tf.sigmoid, 'tanh': tf.tanh, 'relu': tf.nn.relu, 'relu6': tf.nn.relu6,
    'elu': tf.nn.elu, 'softplus': tf.nn.softplus, 'softsign': tf.nn.softsign
    # for detailed intro, go to https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_
}


def getFeature(filename, mode = 'mfcc', feature_len =13, win_step = 0.01, win_len = 0.02):
    (rate,sig)= wav.read(filename)
    feat = calcfeat_delta_delta(sig,rate,
        win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
    feat = preprocessing.scale(feat)
    #feat = np.transpose(feat)
    return feat, len(sig)*(1/rate)

def libri_infer(args, audio_file):
    feat, feat_len = getFeature(audio_file)
    seqLength = feat.shape[0]
    maxTimeSteps = feat.shape[0]
    args.activation = activation_functions_dict[args.activation]

    model = DBiRNN(args, maxTimeSteps)

    print(model.config)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    with tf.Session(graph=model.graph, config=config) as sess:
        # restore from stored model
        ckpt = tf.train.get_checkpoint_state(args.savedir)
        if ckpt and ckpt.model_checkpoint_path:
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored from:' + args.savedir)
            batchInputs = feat[:,np.newaxis,:]
            #batchInputs = feat
            batchSeqLengths = [seqLength]
            feedDict = {model.inputX: batchInputs, model.seqLengths: batchSeqLengths}

            start_time = time.time()
            pre = sess.run([model.predictions], feed_dict=feedDict)

            result = output_to_sequence(pre[0][0])
            log_prob = pre[0][1][0][0]/seqLength
            print('Inference uses ' + str(time.time()-start_time) + 's for '+str(feat_len)+'s audio')

    return result, log_prob

def main():
    args = dict()
    # args['mode'] = 'test'
    # args['level'] = 'cha'
    args['model'] = 'DBiRNN'
    # args['rnncell'] = 'lstm'
    args['num_layer'] = 2
    args['activation'] = 'tanh'
    args['batch_size'] = 1
    args['num_hidden'] = 256
    args['num_feature'] = 39
    args['num_class'] = 29
    args['num_epochs'] = 1
    args['savedir'] = './models/04232130'
    args = dotdict(args)

    libri_infer(args, "./test/out.wav")

if __name__ == '__main__':
    main()
