from utils import dotdict
from libri_inference import libri_infer
from ds_inference import ds_infer

audio = "./test/out.wav"

def main():

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
    args['savedir'] = './models/04232130'
    libri_args = dotdict(args)

    args = dict()
    args['dir_path'] = "./deepspeech-models/"
    args['model'] = dir_path+"output_graph.pb"
    args['lm'] = dir_path+"lm.binary"
    args['trie'] = dir_path+"trie"
    args['alphabet'] = dir_path+"alphabet.txt"
    ds_args = dotdict(args)


    # Main Logic
    libri_result, log_prob = libri_infer(libri_args, audio)
    print("Edge Result:"+libri_result)
    if (log_prob>0.1):
    	print("Result is good enough")
    else:
    	ds_result = ds_infer(ds_args, audio)
    	print("Cloud Result:"+ds_result)

if __name__ == '__main__':
	main()