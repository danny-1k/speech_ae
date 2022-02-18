import os

import torch
import torchaudio
from torch.utils.data import Dataset

import torchaudio.transforms as tatransforms

import numpy as np

import utils.audio_utils as audio_utils


from config import MelSpecConfig


class MelSpec(Dataset):
    def __init__(self,train=True):
        super().__init__()
        self.train = train
        self.mu = MelSpecConfig.TRAIN_MU
        self.std = MelSpecConfig.TRAIN_STD

        if train:
            self.data = [os.path.join('../data/train/mel_spec',f) for f in os.listdir('../data/train/mel_spec')]

            np.random.shuffle(self.data)

            # self.transform = tatransforms.Compose([
            #     self._random_gaussian_noise(),
            # ])

        else:
            self.data = [os.path.join('../data/test/mel_spec',f) for f in os.listdir('../data/test/mel_spec')]
            
            np.random.shuffle(self.data)
            
            # self.transform = tatransforms.Compose([
            # ])


    def _standardize(self,x):
        x = (x-self.mu)/self.std

        return x



    def _random_gaussian_noise(self, audio, snr=None):
        # snr between 0 and 1

        if snr == None:
            snr = torch.rand(1).item()

        noise = torch.normal(mean=torch.zeros_like(audio), std=snr)

        audio = audio+noise

        return audio

    def __getitem__(self, idx):
        x = np.load(self.data[idx])
        
        x = torch.from_numpy(x).unsqueeze(0)

        x = self._standardize(x)


        return x, x

    def __len__(self):
        return len(self.data)


class MFCC(Dataset):
    def __init__(self,MU,STD,train=True,num_train=None,num_test=None):
        super().__init__()
        self.train = train
        self.mu = MU
        self.std = STD

        if train:
            self.data = [os.path.join('../data/train/mfcc',f) for f in os.listdir('../data/train/mfcc')]

            np.random.shuffle(self.data)
            if num_train:
                self.data = self.data[:num_train]

        else:
            self.data = [os.path.join('../data/test/mfcc',f) for f in os.listdir('../data/test/mfcc')]

            np.random.shuffle(self.data)
            if num_test:
                self.data = self.data[:num_test]


    def _standardize(self,x):
        x = (x-self.mu)/self.std

        return x


    def __getitem__(self, idx):
        x = np.load(self.data[idx])
        
        x = torch.from_numpy(x)

        x = self._standardize(x)


        return x, x

    def __len__(self):
        return len(self.data)


class Waveform(Dataset):
    def __init__(self,MU,STD,train=True,num_train=None,num_test=None):
        super().__init__()
        self.train = train
        self.mu = MU
        self.std = STD

        if train:
            self.data = [os.path.join('../data/train/waveform',f) for f in os.listdir('../data/train/waveform')]

            np.random.shuffle(self.data)
            if num_train:
                self.data = self.data[:num_train]

        else:
            self.data = [os.path.join('../data/test/waveform',f) for f in os.listdir('../data/test/waveform')]

            np.random.shuffle(self.data)
            if num_test:
                self.data = self.data[:num_test]


    def _standardize(self,x):
        x = (x-self.mu)/self.std

        return x


    def __getitem__(self, idx):
        x = np.load(self.data[idx])
        
        x = torch.from_numpy(x)

        x = self._standardize(x)

        x = x.view(1,-1)


        return x, x

    def __len__(self):
        return len(self.data)
