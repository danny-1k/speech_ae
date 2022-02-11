import os

import torch
import torchaudio
from torch.utils.data import Dataset

import torchaudio.transforms as tatransforms

import numpy as np

import utils.audio_utils as audio_utils


from config import Config


class SpeechDS(Dataset):
    def __init__(self,train=True):
        super().__init__()
        self.train = train
        self.mu = Config.train_config.TRAIN_MU
        self.std = Config.train_config.TRAIN_STD

        if train:
            self.data = [os.path.join('../data/train',f) for f in os.listdir('../data/train')]

            # self.transform = tatransforms.Compose([
            #     self._random_gaussian_noise(),
            # ])

        else:
            self.data = [os.path.join('../data/test',f) for f in os.listdir('../data/test')]
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
