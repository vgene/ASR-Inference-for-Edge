## Introduction

Automated speech recognition inference for edge devices including Jetson TX1 and Altera Arria 10, etc.

This project modified from following repo. I also used that repo to complete the training process. This modified project is specificly designed for inference process on edge devices. Redundant dependencies are removed.

https://github.com/zzw922cn/Automatic_Speech_Recognition

__Make sure you are using Python 3.5!__

## Preparation

Need to install sox
  sudo apt-get install sox

FOR TX1:

1. Install Lapack: sudo apt-get install liblapack-dev

2. Install requirements: pip install -r requirements-tx1.txt

3. Install TensorFlow: https://github.com/jetsonhacks/installTensorFlowJetsonTX/tree/master/TX1

FOR AMD64:

1. pip install -r requirements.txt

## Usage

Configure args in libri_inference.py, including audio file path and model path.

Run: python3 libri_inference.py

## Procedure

- Flac转wav

    Channels       : 1
    Sample Rate    : 16000
    Precision      : 16-bit
    Bit Rate       : 256k
    Sample Encoding: 16-bit Signed Integer PCM
    Endian Type    : little
    Reverse Nibbles: no
    Reverse Bits   : no

- txt split

- to lower case

- getfeature
```
import scipy.io.wavfile as wav
from sklearn import preprocessing
from calcmfcc import calcfeat_delta_delta

(rate,sig)= wav.read(fullFilename)
# mode = 'mfcc','fbank'
# feature_len =13
# win_step = 0.01
# win_len = 0.02
feat = calcfeat_delta_delta(sig,rate,
    win_length=win_len,win_step=win_step,mode=mode,feature_len=feature_len)
feat = preprocessing.scale(feat)
feat = np.transpose(feat)
#print(feat.shape)
```


