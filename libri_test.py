# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : libri_train.py
# Description  : Training models on LibriSpeech dataset for Automatic Speech Recognition
# ******************************************************

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

flags.DEFINE_string('task', 'libri', 'set task name of this program')
flags.DEFINE_string('train_dataset', 'train-clean-100', 'set the training dataset')
flags.DEFINE_string('dev_dataset', 'dev-clean', 'set the development dataset')
flags.DEFINE_string('test_dataset', 'test-clean', 'set the test dataset')

flags.DEFINE_string('mode', 'test', 'set whether to train, dev or test')

flags.DEFINE_boolean('keep', True, 'set whether to restore a model, when test mode, keep should be set to True')
flags.DEFINE_string('level', 'cha', 'set the task level, phn, cha, or seq2seq, seq2seq will be supported soon')
flags.DEFINE_string('model', 'DBiRNN', 'set the model to use, DBiRNN, BiRNN, ResNet..')
flags.DEFINE_string('rnncell', 'lstm', 'set the rnncell to use, rnn, gru, lstm...')
flags.DEFINE_integer('num_layer', 2, 'set the layers for rnn')
flags.DEFINE_string('activation', 'tanh', 'set the activation to use, sigmoid, tanh, relu, elu...')
flags.DEFINE_string('optimizer', 'adam', 'set the optimizer to use, sgd, adam...')
flags.DEFINE_boolean('layerNormalization', False, 'set whether to apply layer normalization to rnn cell')

flags.DEFINE_integer('batch_size', 64, 'set the batch size')
flags.DEFINE_integer('num_hidden', 256, 'set the hidden size of rnn cell')
flags.DEFINE_integer('num_feature', 39, 'set the size of input feature')
flags.DEFINE_integer('num_classes', 30, 'set the number of output classes')
flags.DEFINE_integer('num_epochs', 1, 'set the number of epochs')
flags.DEFINE_float('lr', 0.0001, 'set the learning rate')
flags.DEFINE_float('dropout_prob', 0.1, 'set probability of dropout')
flags.DEFINE_float('grad_clip', 1, 'set the threshold of gradient clipping, -1 denotes no clipping')
flags.DEFINE_string('datadir', '/home/pony/github/data/libri', 'set the data root directory')
flags.DEFINE_string('logdir', '/home/pony/github/log/libri', 'set the log directory')


FLAGS = flags.FLAGS
task = FLAGS.task

test_dataset = FLAGS.test_dataset

level = FLAGS.level
model_fn = DBiRNN
rnncell = FLAGS.rnncell
num_layer = FLAGS.num_layer

# get activation function and optimizer function
activation_fn = activation_functions_dict[FLAGS.activation]
optimizer_fn = optimizer_functions_dict[FLAGS.optimizer]

batch_size = FLAGS.batch_size
num_hidden = FLAGS.num_hidden
num_feature = FLAGS.num_feature
num_classes = get_num_classes(level)
num_epochs = FLAGS.num_epochs
lr = FLAGS.lr
grad_clip = FLAGS.grad_clip
datadir = FLAGS.datadir

logdir = FLAGS.logdir
savedir = os.path.join(logdir, level, 'save')
resultdir = os.path.join(logdir, level, 'result')
loggingdir = os.path.join(logdir, level, 'logging')
check_path_exists([logdir, savedir, resultdir, loggingdir])

mode = FLAGS.mode
keep = FLAGS.keep
keep_prob = 1-FLAGS.dropout_prob

print('%s mode...'%str(mode))
if mode == 'test' or mode == 'dev':
  batch_size = 10
  num_epochs = 1


def get_data(datadir, test_dataset):
    test_feature_dirs = [os.path.join(os.path.join(datadir, level, test_dataset),
        i, 'feature') for i in os.listdir(os.path.join(datadir, level, test_dataset))]

    test_label_dirs = [os.path.join(os.path.join(datadir, level, test_dataset),
        i, 'label') for i in os.listdir(os.path.join(datadir, level, test_dataset))]
    return test_feature_dirs, test_label_dirs

logfile = os.path.join(loggingdir, str(datetime.datetime.strftime(datetime.datetime.now(),
    '%Y-%m-%d %H:%M:%S') + '.txt').replace(' ', '').replace('/', ''))

class Runner(object):

    def _default_configs(self):
      return {'level': level,
              'rnncell': rnncell,
              'batch_size': batch_size,
              'num_hidden': num_hidden,
              'num_feature': num_feature,
              'num_class': num_classes,
              'num_layer': num_layer,
              'activation': activation_fn,
              'optimizer': optimizer_fn,
              'learning_rate': lr,
              'keep_prob': keep_prob,
              'grad_clip': grad_clip,
            }

    @describe
    def load_data(self, feature_dir, label_dir, mode, level):
        return load_batched_data(feature_dir, label_dir, batch_size, mode, level)


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

if __name__ == '__main__':
  runner = Runner()
  runner.run()
