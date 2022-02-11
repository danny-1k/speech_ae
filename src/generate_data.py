import os
from tqdm import tqdm
import utils.audio_utils as audio_utils
from utils.utilities import list_dirs

from torchaudio.transforms import MelSpectrogram,AmplitudeToDB

import numpy as np

SAVE_DIR = '../data/processed'

SAMPLE_RATE = 16_000
MAX_TIME = 1

N_FFT = 1024
HOP_LEN = 512
N_MELS = 64

# SHOULD_LOG_TRANS = True #Take the log (base 10)


mel_spec = MelSpectrogram(
    sample_rate=SAMPLE_RATE,
    n_fft=N_FFT,
    hop_length=HOP_LEN,
    n_mels=N_MELS
)

power_db = AmplitudeToDB()

idx = len(os.listdir(SAVE_DIR))

if idx!= 0:
    print(f'Warning!! There are files in {SAVE_DIR}')
    print(f'NUMBER OF FILES : {idx}')


for f in tqdm(list_dirs('../data/audio_ds')):
    
    audio,sr = audio_utils.read_audio_file(f)
    audio = audio_utils.convert_to_mono(audio)
    audio = audio_utils.resample(audio,sr,SAMPLE_RATE)

    number_of_splits = audio.shape[-1] // int(SAMPLE_RATE*MAX_TIME)

    if number_of_splits<1:
        audio = audio_utils.cut_to_max_time(audio,SAMPLE_RATE,MAX_TIME)

        audio = mel_spec(audio) #mel spectogram


        audio = power_db(audio)

        audio = audio.numpy()


        # if SHOULD_LOG_TRANS:
        #     audio = np.log10(audio+1e-8) #epsilon to avoid log of 0


        np.save(os.path.join(SAVE_DIR,f'{idx}.npy'),audio)

        idx+=1

    number_of_samples = int(SAMPLE_RATE*MAX_TIME)

    for n in range(number_of_splits):

        if len(audio.shape) == 2:
            audio = audio[0]
        
        slice = audio[n*number_of_samples:][:number_of_samples]
        slice = mel_spec(slice)

        slice = power_db(slice)

        slice = slice.numpy()

        # if SHOULD_LOG_TRANS:
        #     audio = np.log10(audio+1e-8) #epsilon to avoid log of 0


        np.save(os.path.join(SAVE_DIR,f'{idx}.npy'),slice)

        idx+=1