import python_bind
import pickle
import numpy as np

audio = "./test/1069-133709-0000.wav"

def preprocess_audio(audio_path):
    import subprocess
    import scipy.io.wavfile as wav
    fs, audio = wav.read(audio_path)
    if fs != 16000:
        if fs < 16000:
            print('Warning: original sample rate (%d) is lower than 16kHz. Up-sampling might produce erratic speech recognition.' % (fs), file=sys.stderr)
        fs, audio = convert_samplerate(audio_path)

    sox_cmd = 'sox {} --type raw --bits 16 --channels 1 --rate 16000 - '.format(audio_path)
    try:
        p = subprocess.Popen(sox_cmd.split(),
                             stderr=subprocess.PIPE, stdout=subprocess.PIPE)
        output, err = p.communicate()

        if p.returncode:
            raise RuntimeError('SoX returned non-zero status: {}'.format(err))

    except OSError as e:
        raise OSError('SoX not found, use 16kHz files or install it: ', e)

    audio = np.fromstring(output, dtype=np.int16)
    return 16000, audio

def get_args_edge():
    from utils import dotdict, activation_functions_dict
    args = dict()
    # args['mode'] = 'test'
    # args['level'] = 'cha'
    args['model'] = 'DBiRNN'
    # args['rnncell'] = 'lstm'
    args['num_layer'] = 2
    args['activation'] = activation_functions_dict['tanh']
    args['batch_size'] = 1
    args['num_hidden'] = 256
    args['num_feature'] = 39
    args['num_class'] = 29
    args['num_epochs'] = 1
    args['savedir'] = './models/04262030'
    return dotdict(args)
    
def get_args_cloud():
    from utils import dotdict
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
