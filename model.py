import torch
import torch.nn as nn


class WaveNet(nn.Module):
    def __init__(self,input_size):
        super().__init__()
        self.input_size = input_size

        self.encoder = nn.Sequential(
            nn.Linear(input_size,2048),
            nn.ReLU(),
            nn.Linear(2048,600),
        )

        self.decoder = nn.Sequential(
            nn.Linear(600,2048),
            nn.ReLU(),
            nn.Linear(2048,input_size),
        )


    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x