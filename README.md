


# Procedure

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

- 转小写 

- 转feature
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


