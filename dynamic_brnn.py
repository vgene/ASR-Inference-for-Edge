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
                             seqLengths,
                             time_major=True):
    hid_input = inputX
    for i in range(args.num_layer):
        scope = 'DBRNN_' + str(i + 1)
        forward_cell = cell_fn(args.num_hidden, activation=args.activation)
        backward_cell = cell_fn(args.num_hidden, activation=args.activation)
        # tensor of shape: [max_time, batch_size, input_size]
        outputs, output_states = bidirectional_dynamic_rnn(forward_cell, backward_cell,
                                                           inputs=hid_input,
                                                           dtype=tf.float32,
                                                           sequence_length=seqLengths,
                                                           time_major=True,
                                                           scope=scope)
        # forward output, backward ouput
        # tensor of shape: [max_time, batch_size, input_size]
        output_fw, output_bw = outputs
        # forward states, backward states
        output_state_fw, output_state_bw = output_states
        # output_fb = tf.concat(2, [output_fw, output_bw])
        output_fb = tf.stack([output_fw, output_bw], 2, name='output_fb')
        #output_fb = tf.concat([output_fw, output_bw], 2)
        output_fb = tf.Print(output_fb, [output_fb.get_shape()], message='output_fb:')
        #shape = output_fb.get_shape().as_list()
        #output_fb = tf.reshape(output_fb, [shape[0], shape[1], 2, int(shape[2] / 2)])
        #output_fb = tf.Print(output_fb, [output_fb.get_shape()], message='after reshape:')
        hidden = tf.reduce_sum(output_fb, 2)
        # hidden = dropout(hidden, args.keep_prob, (args.mode == 'train')) #Apr 19

        if i != args.num_layer - 1:
            hid_input = hidden
        else:
            outputXrs = tf.reshape(hidden, [-1, args.num_hidden])
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
                                         shape=(maxTimeSteps, args.batch_size, args.num_feature), name='inputX')  # [maxL,32,39]
            inputXrs = tf.reshape(self.inputX, [-1, args.num_feature])
            self.inputList = tf.split(inputXrs, maxTimeSteps, 0)  # convert inputXrs from [32*maxL,39] to [32,maxL,39]
            self.seqLengths = tf.placeholder(tf.int32, shape=(args.batch_size), name='seqLengths')
            self.config = {'name': args.model,
                           'rnncell': self.cell_fn,
                           'num_layer': args.num_layer,
                           'num_hidden': args.num_hidden,
                           'num_class': args.num_class,
                           'activation': args.activation,
                           'batch size': args.batch_size}

            fbHrs = build_multi_dynamic_brnn(self.args, maxTimeSteps, self.inputX, self.cell_fn, self.seqLengths)
            with tf.name_scope('fc-layer'):
                with tf.variable_scope('fc'):
                    weightsClasses = tf.Variable(
                        tf.truncated_normal([args.num_hidden, args.num_class], name='weightsClasses'))
                    biasesClasses = tf.Variable(tf.zeros([args.num_class]), name='biasesClasses')
                    logits = [tf.matmul(t, weightsClasses) + biasesClasses for t in fbHrs]
            logits3d = tf.stack(logits)
            # self.loss = tf.reduce_mean(tf.nn.ctc_loss(self.targetY, logits3d, self.seqLengths)) #Apr 19

            self.predictions = tf.nn.ctc_beam_search_decoder(logits3d, self.seqLengths, merge_repeated=False, beam_width=500)
            # if args.level == 'cha':
            #     self.errorRate = tf.reduce_sum(tf.edit_distance(self.predictions, self.targetY, normalize=True))
            self.initial_op = tf.global_variables_initializer()
            self.saver = tf.train.Saver(tf.global_variables(), max_to_keep=5, keep_checkpoint_every_n_hours=1)
