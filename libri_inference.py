import scipy.io.wavfile as wav
from sklearn import preprocessing
from calcmfcc import calcfeat_delta_delta

import time
import datetime
import os
from six.moves import cPickle
from functools import wraps
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

from speechvalley.utils import load_batched_data, describe, output_to_sequence, list_dirs, logging, count_params, get_num_classes, check_path_exists, dotdict, activation_functions_dict, optimizer_functions_dict
from speechvalley.models import DBiRNN

from tensorflow.python.platform import flags
from tensorflow.python.platform import app

mode = 'mfcc'
feature_len =13
win_step = 0.01
win_len = 0.02

(rate,sig)= wav.read(fullFilename)
feat = calcfeat_delta_delta(sig,rate,
    win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
feat = preprocessing.scale(feat)
feat = np.transpose(feat)


def run(self):
    # load data
    args_dict = self._default_configs()
    args = dotdict(args_dict)
    feature_dirs, label_dirs = get_data(datadir, test_dataset)
    batchedData, maxTimeSteps, totalN = self.load_data(feature_dirs[0], label_dirs[0], mode, level)
    model = model_fn(args, maxTimeSteps)

    ## shuffle feature_dir and label_dir by same order
    FL_pair = list(zip(feature_dirs, label_dirs))
    random.shuffle(FL_pair)
    feature_dirs, label_dirs = zip(*FL_pair)

    for feature_dir, label_dir in zip(feature_dirs, label_dirs):
        id_dir = feature_dirs.index(feature_dir)
        print('dir id:{}'.format(id_dir))
        batchedData, maxTimeSteps, totalN = self.load_data(feature_dir, label_dir, mode, level)
        model = model_fn(args, maxTimeSteps)
        num_params = count_params(model, mode='trainable')
        all_num_params = count_params(model, mode='all')
        model.config['trainable params'] = num_params
        model.config['all params'] = all_num_params
        print(model.config)
        with tf.Session(graph=model.graph) as sess:
            # restore from stored model
            if keep == True:
                ckpt = tf.train.get_checkpoint_state(savedir)
                if ckpt and ckpt.model_checkpoint_path:
                    model.saver.restore(sess, ckpt.model_checkpoint_path)
                    print('Model restored from:' + savedir)
            else:
                print('Initializing')
                sess.run(model.initial_op)

            for epoch in range(num_epochs):
                ## training
                start = time.time()
                if mode == 'train':
                    print('Epoch {} ...'.format(epoch + 1))

                batchErrors = np.zeros(len(batchedData))
                batchRandIxs = np.random.permutation(len(batchedData))

                for batch, batchOrigI in enumerate(batchRandIxs):
                    batchInputs, batchTargetSparse, batchSeqLengths = batchedData[batchOrigI]
                    batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
                    feedDict = {model.inputX: batchInputs, model.targetIxs: batchTargetIxs,
                                model.targetVals: batchTargetVals, model.targetShape: batchTargetShape,
                                model.seqLengths: batchSeqLengths}

                    if level == 'cha':
                        l, pre, y, er = sess.run([model.loss, model.predictions,
                            model.targetY, model.errorRate], feed_dict=feedDict)
                        batchErrors[batch] = er
                        print('\n{} mode, total:{},subdir:{}/{},batch:{}/{},test loss={:.3f},mean test CER={:.3f}\n'.format(
                            level, totalN, id_dir+1, len(feature_dirs), batch+1, len(batchRandIxs), l, er/batch_size))

                    # NOTE:
                    if er / batch_size == 1.0:
                        break

                    if batch % 20 == 0:
                        print('Truth:\n' + output_to_sequence(y, type=level))
                        print('Output:\n' + output_to_sequence(pre, type=level))

                if mode=='test' or mode=='dev':
                    with open(os.path.join(resultdir, level + '_result.txt'), 'a') as result:
                        result.write(output_to_sequence(y, type=level) + '\n')
                        result.write(output_to_sequence(pre, type=level) + '\n')
                        result.write('\n')
                    epochER = batchErrors.sum() / totalN
                    print(' test error rate:', epochER)
                    logging(model, logfile, epochER, mode=mode)
