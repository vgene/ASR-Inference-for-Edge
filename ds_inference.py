# -*- coding: utf-8 -*-
from __future__ import absolute_import, division, print_function
from timeit import default_timer as timer

import sys
import numpy as np
import python_bind
import pickle
from utils import dotdict, preprocess_audio
from deepspeech.model import Model

# Beam width used in the CTC decoder when building candidate transcriptions
# The alpha hyperparameter of the CTC decoder. Language Model weight
# The beta hyperparameter of the CTC decoder. Word insertion weight (penalty)
# Valid word insertion weight. This is used to lessen the word insertion penalty
# when the inserted word is part of the vocabulary
# Number of MFCC features to use
# Size of the context window used for producing timesteps in the input vector
def load_model(args):
    BEAM_WIDTH = 500
    LM_WEIGHT = 1.75
    WORD_COUNT_WEIGHT = 1.00
    VALID_WORD_COUNT_WEIGHT = 1.00
    N_FEATURES = 26
    N_CONTEXT = 9

    print('Loading model from file %s' % (args.model), file=sys.stderr)
    model_load_start = timer()
    ds = Model(args.model, N_FEATURES, N_CONTEXT, args.alphabet, BEAM_WIDTH)
    model_load_end = timer() - model_load_start
    print('Loaded model in %0.3fs.' % (model_load_end), file=sys.stderr)

    if args.lm and args.trie:
        print('Loading language model from files %s %s' % (args.lm, args.trie), file=sys.stderr)
        lm_load_start = timer()
        ds.enableDecoderWithLM(args.alphabet, args.lm, args.trie, LM_WEIGHT,
                               WORD_COUNT_WEIGHT, VALID_WORD_COUNT_WEIGHT)
        lm_load_end = timer() - lm_load_start
        print('Loaded language model in %0.3fs.' % (lm_load_end), file=sys.stderr)

    return ds

def ds_server(args=None):
    if ars is None:
        args = get_args()

    ds = load_model(args)

    print('Start Deep Speech Server.')
    # Init Server
    python_bind.init_cloud_server()

    while True:
        try:
            (fs, audio) = pickle.loads(python_bind.recv())
            inference_start = timer()
            result = ds.stt(audio, fs)
            print(result)
            inference_end = timer() - inference_start
            audio_length = len(audio)*(1/fs)
            print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)
            pld = pickle.dumps(result, 2)
            python_bind.send(pld)
        except Exception as e:
            import traceback
            traceback.print_exc()
            break

def ds_infer(args, audio_path):
    ds = load_model(args)
    fs, audio = preprocess_audio(audio_path)
    audio_length = len(audio) * ( 1 / 16000)

    inference_start = timer()
    result = ds.stt(audio, fs)
    print(result)
    inference_end = timer() - inference_start
    print('Inference took %0.3fs for %0.3fs audio file.' % (inference_end, audio_length), file=sys.stderr)
    return result

def get_args():
    args = dict()
    dir_path = "./deepspeech-models/"
    args['model'] = dir_path+"output_graph.pb"
    args['lm'] = dir_path+"lm.binary"
    args['trie'] = dir_path+"trie"
    args['alphabet'] = dir_path+"alphabet.txt"
    ds_args = dotdict(args)
    return ds_args

if __name__ == '__main__':
    #args = get_args()
    #audio = "./test/out.wav"
    #ds_infer(args, audio)
    ds_server()
