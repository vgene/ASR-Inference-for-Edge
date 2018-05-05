import time
from functools import wraps
import subprocess
import scipy.io.wavfile as wav
import numpy as np
from sklearn import preprocessing
from calcmfcc import calcfeat_delta_delta

class dotdict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__

def describe(func):
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

def getFeature(filename, mode = 'mfcc', feature_len =13, win_step = 0.01, win_len = 0.02):
    (rate,sig)= wav.read(filename)
    feat = calcfeat_delta_delta(sig,rate,
        win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
    feat = preprocessing.scale(feat)
    #feat = np.transpose(feat)
    return feat, len(sig)*(1/rate)

def preprocess_audio(audio_path):
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
