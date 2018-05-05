from timeit import default_timer as timer
import pickle
import numpy as np
import pprint

import python_bind
from utils import preprocess_audio, dotdict
from libri_inference import libri_infer
import argparse

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

    t0 = timer()
    (fs, audio) = preprocess_audio(audio)
    t1 = timer()

    data = pickle.dumps((fs,audio), 2)
    python_bind.send(data)
    t2 = timer()

    ds_result = pickle.loads(python_bind.recv())
    t3 = timer()

    return {"cloud_result":ds_result, "cloud_preprocess_time": t1 - t0,
            "cloud_transfer_time":t2 - t1, "cloud_receive_time":t3 - t2}

def main():
    parser = argparse.ArgumentParser(description='Edge Computing for ASR - Test Run')
    parser.add_argument('audio_file', type=str,
                        help='Path to the audio file to run (WAV format)')
    parser.add_argument('--ref_file', nargs="?", type=str,
                        help='Path to the audio reference file')
    parser.add_argument('--cloud_ip', type=str, nargs='?',
                        help='IP of cloud server')
    parser.add_argument('--cloud_port', type=int, nargs='?',
                        help='Port of cloud server')
    args = parser.parse_args()

    # TODO: set IP and Port
    # python_bind.set(IP)
    #

    # TODO: check audio file!
    # TODO: check ref file!

    # Main Logic
    edge_args = get_args_edge()
    cloud_args = get_args_cloud()

    edge_start_time = timer()
    edge_results = get_results_edge(edge_args, args.audio_file)
    edge_total_time = timer() - edge_start_time

    print("Edge Result:\n"+edge_results['edge_result'])
    print("Log Prob:"+str(edge_results['edge_log_prob']))
    if (edge_results['edge_log_prob']>0.1):
        print("Result is good enough")
    else:
        cloud_start_time = timer()
        cloud_results = get_results_cloud(cloud_args, args.audio_file)
        cloud_total_time = timer() - cloud_start_time
        print("Cloud Result:\n"+cloud_results['cloud_result'])

    # Calculate WER
    if args.ref_file:
        from wer import wer
        with open(args.ref_file, 'r') as ref_text:
            ref = ref_text.readlines()[0]

            edge_wer, edge_match_count, ref_token_count = wer(ref, edge_results['edge_result'])
            cloud_wer, cloud_match_count, ref_token_count = wer(ref, cloud_results['cloud_result'])
            print('Edge WER: {:10.3%} ({:4d} / {:4d})'.format(edge_wer, edge_match_count, ref_token_count))
            print('Edge WER: {:10.3%} ({:4d} / {:4d})'.format(cloud_wer, cloud_match_count, ref_token_count))

    times = {**edge_results, **cloud_results}
    times.pop('cloud_result', None)
    times.pop('edge_result', None)
    times.pop('edge_log_prob', None)
    print(edge_total_time)
    print(cloud_total_time)
    pprint.pprint(times)

if __name__ == '__main__':
    main()
