from timeit import default_timer as timer
import os
import random

import numpy as np
import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc

# from speechvalley.utils import count_params
from dynamic_brnn_infer import DBiRNN
from utils import dotdict, describe, output_to_sequence, getFeature

ADD_TRACE = True
PROFILE_TRACE = True

activation_functions_dict = {
    'sigmoid': tf.sigmoid, 'tanh': tf.tanh, 'relu': tf.nn.relu, 'relu6': tf.nn.relu6,
    'elu': tf.nn.elu, 'softplus': tf.nn.softplus, 'softsign': tf.nn.softsign
    # for detailed intro, go to https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_
}

def libri_infer(args, audio_file):
    t0 = timer()
    feat, feat_len, audio_len = getFeature(audio_file)
    t1 = timer()

    seqLength = feat_len
    maxTimeSteps = feat.shape[0]
    args.activation = activation_functions_dict[args.activation]

    model = DBiRNN(args, maxTimeSteps)
    t2 = timer()

    print(model.config)
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    #config.graph_options.optimizer_options.global_jit_level = tf.OptimizerOptions.ON_1

    with tf.Session(graph=model.graph, config=config) as sess:
        # restore from stored model
        ckpt = tf.train.get_checkpoint_state(args.savedir)
        if ckpt and ckpt.model_checkpoint_path:
            model.saver.restore(sess, ckpt.model_checkpoint_path)
            print('Model restored from:' + args.savedir)
            batchInputs = feat[:,np.newaxis,:]
            #batchInputs = feat
            #batchSeqLengths = [seqLength]
            batchSeqLengths = seqLength
            feedDict = {model.inputX: batchInputs, model.seqLengths: batchSeqLengths}
            t3 = timer()

            pre = sess.run([model.predictions], feed_dict=feedDict)
            result = output_to_sequence(pre[0][0])
            log_prob = pre[0][1][0][0]/seqLength
            t4 = timer()

    return {"result":result, "log_prob":log_prob, "preprocess_time":t1-t0,
            "build_model_time":t2-t1, "start_session_time":t3-t2,
            "infer_time":t4-t3}


def load_graph(frozen_graph_filename):
    # We load the protobuf file from the disk and parse it to retrieve the
    # unserialized graph_def
    t0 = timer()
    with tf.gfile.FastGFile(frozen_graph_filename, "rb") as f:
        graph_def = tf.GraphDef()
        graph_def.ParseFromString(f.read())
    t1 = timer()
    # Then, we import the graph_def into a new Graph and returns it
    with tf.Graph().as_default() as graph:
        # The name var will prefix every op/nodes in your graph
        # Since we load everything in a new graph, this is not needed
        dir(tf.contrib)
        tf.import_graph_def(graph_def, name='infer')
    t2 = timer()

    print("Load Model Time:"+str(t1-t0))
    print("Load Model Parsing Time:"+str(t2-t1))
    return graph


def libri_infer_from_freeze(args, audio_file):
    t0 = timer()
    feat, feat_len, audio_len = getFeature(audio_file, maxLength=2000)
    t1 = timer()
    graph = load_graph(os.path.join(args.savedir,'frozen_model_1.pb'))
    t2 = timer()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    run_metadata = tf.RunMetadata()
    with tf.Session(graph=graph, config=config) as sess:
        # restore from stored model
        #for op in graph.get_operations():
        #    print(op.name)
        batchInputs = feat[:,np.newaxis,:]
        #batchInputs = feat
        batchSeqLengths = [feat_len]
        feedDict = {'infer/inputX:0': batchInputs, 'infer/seqLengths:0': batchSeqLengths}
        t3 = timer()

        logits3d = graph.get_tensor_by_name("infer/stack:0")
        predictions = tf.nn.ctc_beam_search_decoder(logits3d, batchSeqLengths, merge_repeated=False, beam_width=100, top_paths=1)

        if ADD_TRACE:
            pre = sess.run(predictions, feed_dict=feedDict,
                    options=tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE), run_metadata=run_metadata)
        else:
            pre = sess.run(predictions, feed_dict=feedDict)

        result = output_to_sequence(pre[0])
        log_prob = pre[1][0][0]/feat_len

        t4 = timer()

        if ADD_TRACE:
            LOGDIR='./log'
            infer_writer = tf.summary.FileWriter(LOGDIR)
            infer_writer.add_graph(sess.graph)
            infer_writer.add_run_metadata(run_metadata,'infer')

    if PROFILE_TRACE:
        ProfileOptionBuilder = tf.profiler.ProfileOptionBuilder
        opts = ProfileOptionBuilder(ProfileOptionBuilder.time_and_memory()
            ).with_node_names(show_name_regexes=['.*']).build()

        tf.profiler.profile(
            tf.get_default_graph(),
            run_meta=run_metadata,
            cmd='code',
            options=opts)
       
        # Print to stdout an analysis of the memory usage and the timing information
        # broken down by operation types.
        tf.profiler.profile(
            tf.get_default_graph(),
            run_meta=run_metadata,
            cmd='op',
            options=tf.profiler.ProfileOptionBuilder.time_and_memory())

    return {"result":result, "log_prob":log_prob, "preprocess_time":t1-t0,
            "build_model_time":t2-t1, "start_session_time":t3-t2,
            "infer_time":t4-t3}

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
