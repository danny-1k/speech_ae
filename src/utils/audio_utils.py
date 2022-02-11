import torch
import torchaudio

import numpy as np


def expand_array(arr,new_length):
    # https://stackoverflow.com/questions/66934748/how-to-stretch-an-array-to-a-new-length-while-keeping-same-value-distribution
    if len(arr) < new_length:
        array_len = len(arr)
        arr = np.interp(np.linspace(0, array_len - 1, num=new_length), np.arange(array_len), arr)

    return arr


def cut_to_max_time(audio,sample_rate,max_time):
    if audio.shape[-1]/sample_rate > max_time:
        audio = audio[:,:int(sample_rate*max_time)]

    elif audio.shape[-1]/sample_rate < max_time:
        if len(audio.shape) > 1:
            audio = audio[0]
        audio = torch.from_numpy(expand_array(audio.numpy(),int(sample_rate*max_time)))

    return audio


def read_audio_file(f):
    audio, sr = torchaudio.load(f)
    return audio, sr


def resample(audio,original_sample_rate,new_sample_rate):
    audio = torchaudio.transforms.Resample(
        orig_freq=original_sample_rate, new_freq=new_sample_rate)(audio)
        
    return audio


def convert_to_mono(audio):
    if audio.shape[0] == 2:
        audio = audio.mean(axis=0)

    return audio

