import python_bind
import pickle
import numpy as np
from utils import preprocess_audio
from utils import dotdict

audio = "./test/out.wav"

def get_args_edge():
    args = dict()
    args['model'] = 'DBiRNN'
    args['num_layer'] = 2
    args['activation'] = 'tanh'
    args['batch_size'] = 1
    args['num_hidden'] = 256
    args['num_feature'] = 39
    args['num_class'] = 29
    args['num_epochs'] = 1
    args['savedir'] = './models/04262030'
    return dotdict(args)

def get_args_cloud():
    args = dict()
    dir_path = "./deepspeech-models/"
    args['model'] = dir_path+"output_graph.pb"
    args['lm'] = dir_path+"lm.binary"
    args['trie'] = dir_path+"trie"
    args['alphabet'] = dir_path+"alphabet.txt"
    return dotdict(args)

def get_results_edge(args, audio):
    from libri_inference import libri_infer

    libri_result, log_prob = libri_infer(args, audio)

    return libri_result, log_prob

def get_results_cloud(args, audio):
    python_bind.init_edge()
    (fs, audio) = preprocess_audio(audio)
    data = pickle.dumps((fs,audio), 2)
    python_bind.send(data)
    ds_result = pickle.loads(python_bind.recv())
    return ds_result

def main():
    # Main Logic
    edge_args = get_args_edge()
    cloud_args = get_args_cloud()

    edge_result, log_prob = get_results_edge(edge_args, audio)
    print("Edge Result:\n"+edge_result)
    print("Log Prob:"+str(log_prob))
    if (log_prob>0.1):
    	print("Result is good enough")
    else:
    	cloud_result = get_results_cloud(cloud_args, audio)
    	print("Cloud Result:\n"+cloud_result)

if __name__ == '__main__':
    main()
