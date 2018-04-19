import time
from functools import wraps
import tensorflow as tf

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def describe(func):
    ''' wrap function,to add some descriptions for function and its running time
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        print(func.__name__+'...')
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        print(str(func.__name__+' in '+ str(end-start)+' s'))
        return result
    return wrapper

def output_to_sequence(lmt):
    indexes = lmt[0][1]
    seq = []
    for ind in indexes:
        if ind == 0:
            seq.append(' ')
        elif ind == 27:
            seq.append("'")
        elif ind == 28:
            pass
        else:
            seq.append(chr(ind+96))
    seq = ''.join(seq)
    return seq

activation_functions_dict = {
    'sigmoid': tf.sigmoid, 'tanh': tf.tanh, 'relu': tf.nn.relu, 'relu6': tf.nn.relu6,
    'elu': tf.nn.elu, 'softplus': tf.nn.softplus, 'softsign': tf.nn.softsign
    # for detailed intro, go to https://www.tensorflow.org/versions/r0.12/api_docs/python/nn/activation_functions_
    }
