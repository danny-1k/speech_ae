import sys
# print(sys.path)
sys.path.append('src')
import torch

from random import sample

# from src.data import SpeechDS

from scipy.io import wavfile

import matplotlib.pyplot as plt

from src.models.convae import ConvAE


import src.utils.audio_utils as audio_utils

import numpy as np

import math

import librosa

from torchaudio.transforms import MelSpectrogram,AmplitudeToDB,InverseMelScale
from torchaudio.functional import DB_to_amplitude


SR = 16_000





N_FFT = 1024
HOP_LEN = 512
N_MELS = 64

mel_spec = MelSpectrogram(
    sample_rate=SR,
    n_fft=N_FFT,
    hop_length=HOP_LEN,
    n_mels=N_MELS
)

power_db = AmplitudeToDB()

inverse_mel = InverseMelScale(
    sample_rate=SR,
    n_stft=513,
    n_mels=N_MELS
)


MU = -13.7342
STD = 16.1290


# with torch.no_grad():

net = ConvAE()
net.load_model_('models')
# net.load_state_dict(torch.load('../models/WaveNet.pt'))
# net.eval()



audio,sr = audio_utils.read_audio_file('test_audio.wav')

audio = audio_utils.convert_to_mono(audio)

print(audio.shape, 'after mono')

# plt.plot(audio[0])
# plt.show()
audio = audio_utils.resample(audio,sr,SR) 
   


out = []
truth = []

number_of_slices = int(math.floor((audio.shape[-1]/int(1*SR))+.5))

print(number_of_slices,'number of slices')

for i in range(number_of_slices):
    if len(audio.shape) > 1:
        audio = audio[0]

    
    current = audio[i*int(1*SR):][:int(1*SR)]
    current = audio_utils.cut_to_max_time(current,SR,1)

    current = power_db(mel_spec(current.float()))

    current = (current-MU)/STD


    # print(current.shape)



    current = current.unsqueeze(0).unsqueeze(0)

    reconstructed = net(current) #(1,1,n,m) 
    # print(reconstructed.shape,'recon')
    reconstructed = reconstructed.squeeze() #(1,n,m)
    # print(reconstructed.shape,'squeeze')

    reconstructed = (reconstructed*STD)+MU


    reconstructed = DB_to_amplitude(reconstructed,1,1)
    reconstructed = reconstructed.detach().numpy()
    # reconstructed = inverse_mel(reconstructed)

    reconstructed = librosa.feature.inverse.mel_to_audio(reconstructed,sr=SR,
                                                            n_fft = N_FFT,
                                                            hop_length=HOP_LEN,
                                                            power=1,


                                                        )

    # print(reconstructed.shape)

    # plt.imshow(reconstructed)
    # plt.show()
    # plt.imshow(DB_to_amplitude(current[0][0],1,1))

    # plt.show()

    print(current.shape,'done')

    out.append(reconstructed)
        
    truth.append(current.numpy())

    
out = np.hstack(out)
truth = np.hstack(truth)

# print(out.shape, 'out shape')




# audio = ds.cut_to_max_time(audio)

# plt.plot(audio[0])
# plt.show()

# from model import WaveNet

# net = WaveNet(4_000)
# net.load_state_dict(torch.load('models/WaveNet.pt'))
# net.eval()

# reconstructed = net(audio)

# plt.plot(audio[0],label='real')


# plt.plot(reconstructed[0].detach().numpy(),label='reconstructed')

# plt.legend()

# plt.show()


# out = out/out.max()

plt.plot(out)
plt.show()

wavfile.write('reconstructed.wav',SR,out)
# wavfile.write('truth.wav',SR,truth)