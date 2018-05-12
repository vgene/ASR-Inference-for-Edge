# encoding: utf-8
# ******************************************************
# Author       : zzw922cn
# Last modified: 2017-12-09 11:00
# Email        : zzw922cn@gmail.com
# Filename     : dynamic_brnn.py
# Description  : Dynamic Bidirectional RNN model for Automatic Speech Recognition
# ******************************************************

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.rnn import bidirectional_dynamic_rnn

# from speechvalley.utils import describe, dropout
from utils import describe

def build_multi_dynamic_brnn(args,
                             maxTimeSteps,
                             inputX,
                             cell_fn,
                             seqLength,
                             time_major=True):
    hid_input = inputX
    for i in range(args.num_layer):
        scope = 'DBRNN_' + str(i + 1)
        forward_cell = cell_fn(args.num_hidden, activation=args.activation)
        backward_cell = cell_fn(args.num_hidden, activation=args.activation)
        # tensor of shape: [max_time, batch_size, input_size]
        outputs, _ = bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                           inputs=hid_input,
                                                           dtype=tf.float32,
                                                           sequence_length=seqLength,
                                                           time_major=time_major,
                                                           scope=scope)
        # forward output, backward ouput
        # tensor of shape: [max_time, batch_size, input_size]
        output_fw, output_bw = outputs
        output_fb = tf.stack([output_fw, output_bw], 2, name='output_fb')
        hidden = tf.reduce_sum(output_fb, 2)

        if i != args.num_layer - 1:
            hid_input = hidden
        else:
            outputXrs = tf.reshape(hidden, [-1, args.num_hidden])
            outputXrs = tf.Print(outputXrs, [outputXrs.get_shape()], message='shape of output xrs')
            # output_list = tf.split(0, maxTimeSteps, outputXrs)
            output_list = tf.split(outputXrs, maxTimeSteps, 0)
            fbHrs = [tf.reshape(t, [args.batch_size, args.num_hidden]) for t in output_list]

    return fbHrs


class DBiRNN(object):
    def __init__(self, args, maxTimeSteps):
        self.args = args
        self.maxTimeSteps = maxTimeSteps
        self.cell_fn = tf.contrib.rnn.BasicLSTMCell
        self.build_graph(args, maxTimeSteps)

    @describe
    def build_graph(self, args, maxTimeSteps):
        self.graph = tf.Graph()
        with self.graph.as_default():
            self.inputX = tf.placeholder(tf.float32,
                                         shape=(maxTimeSteps, 1, args.num_feature), name='inputX')  # [maxL, 1, 39]
            self.seqLength = tf.placeholder(tf.int32, name='seqLengths') #batch_size=1
            self.config = {'name': args.model,
                           'rnncell': self.cell_fn,
                           'num_layer': args.num_layer,
                           'num_hidden': args.num_hidden,
                           'num_class': args.num_class,
                           'activation': args.activation,
                           'batch size': 1}

            fbHrs = build_multi_dynamic_brnn(self.args, self.maxTimeSteps, self.inputX, self.cell_fn, self.seqLength)
            with tf.name_scope('fc-layer'):
                with tf.variable_scope('fc'):
                    weightsClasses = tf.Variable(
                        tf.truncated_normal([args.num_hidden, args.num_class], name='weightsClasses'))
                    biasesClasses = tf.Variable(tf.zeros([args.num_class]), name='biasesClasses')
                    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in fbHrs]
            logits3d = tf.stack(logits)
            # self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targetY, logits3d, self.seqLengths)) #Apr 19

            # self.predictions = tf.nn.ctc_beam_search_decoder(logits3d, self.seqLengths, merge_repeated=False, beam_width=500)
            # if args.level == 'cha':
            #     self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))
            # self.initial_op = tf.global_variables_initializer()
            # self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
