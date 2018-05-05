import python_bind
import pickle
import numpy as np
from utils import preprocess_audio
from utils import dotdict
import time
from libri_inference import libri_infer

audio = "./test/1069-133709-0000.wav"

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
    libri_results = libri_infer(args, audio)
    return {"edge_result":libri_results['result'],
            "edge_log_prob":libri_results['log_prob'],
            "edge_preprocess_time":libri_results['preprocess_time'],
            "edge_build_model_time":libri_results['build_model_time'],
            "edge_start_session_time":libri_results['start_session_time'],
            "edge_infer_time":libri_results['infer_time']}

# Args is no use
def get_results_cloud(args, audio):
    python_bind.init_edge()

    start_time = time.time()
    (fs, audio) = preprocess_audio(audio)
    preprocess_time = time.time() - start_time

    data = pickle.dumps((fs,audio), 2)
    python_bind.send(data)
    transfer_time = time.time() - preprocess_time

    ds_result = pickle.loads(python_bind.recv())
    receive_time = time.time() - transfer_time

    return {"cloud_result":ds_result, "cloud_preprocess_time": preprocess_time,
            "cloud_transfer_time":transfer_time, "cloud_receive_time":receive_time}

def main():
    # Main Logic
    edge_args = get_args_edge()
    cloud_args = get_args_cloud()

    edge_start_time = time.time()
    edge_results = get_results_edge(edge_args, audio)
    edge_total_time = time.time() - edge_start_time

    print("Edge Result:\n"+edge_result)
    print("Log Prob:"+str(log_prob))
    if (edge_results['log_prob']>0.1):
    	print("Result is good enough")
    else:
        cloud_start_time = time.time()
    	cloud_results = get_results_cloud(cloud_args, audio)
        cloud_total_time = time.time() - cloud_start_time
    	print("Cloud Result:\n"+cloud_results['cloud_result'])

    times = {**edge_results, **cloud_results}
    times.pop('cloud_result', None)
    times.pop('edge_result', None)
    print(times)

if __name__ == '__main__':
    main()
