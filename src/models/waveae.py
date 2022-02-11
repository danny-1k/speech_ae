import os

import torch
import torch.nn as nn


class WaveNet(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size,4000),
            nn.ReLU(),
            nn.Linear(4000,2000),
            nn.ReLU(),
            nn.Linear(2000,1000),
            nn.ReLU(),
            nn.Linear(1000,600)
        )

        self.decoder = nn.Sequential(
            nn.Linear(600,1000),
            nn.ReLU(),
            nn.Linear(1000,2000),
            nn.ReLU(),
            nn.Linear(2000,4000),
            nn.ReLU(),
            nn.Linear(4000,input_size)
        )


    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


    def save_model(self,dir):
        torch.save(self.state_dict(),os.path.join(dir,'WaveNet.pt'))


    def load_model_(self,dir):
        self.load_state_dict(torch.load(os.path.join(dir,'WaveNet.pt')))