import os

import torch
import torch.nn as nn


class ConvAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Conv2d(in_channels=1,out_channels=16,kernel_size=3,stride=1,padding=1), # (16,64,32)
            nn.MaxPool2d(2,2), # (16,32,16)
            nn.ReLU(), 

            nn.Conv2d(in_channels=16,out_channels=32,kernel_size=3,stride=1,padding=1),  # (32, 32,16 )
            nn.MaxPool2d(2,2), # (32,16,8)
            nn.ReLU(),

            nn.Conv2d(in_channels=32,out_channels=32,kernel_size=3,stride=1,padding=1), # (32,16,8)
            nn.MaxPool2d(2,2), # (8,4)
            nn.ReLU(),
        )

        self.decoder = nn.Sequential(

            nn.ConvTranspose2d(in_channels=32,out_channels=32,kernel_size=2,stride=2), # (32,16,8)
            nn.ReLU(), # (32,16,8)

            nn.ConvTranspose2d(in_channels=32,out_channels=16,kernel_size=2,stride=2), # (16,32,16)
            nn.ReLU(),

            nn.ConvTranspose2d(in_channels=16,out_channels=1,kernel_size=2,stride=2), # (1,64,32)
        )


    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


    def save_model(self,dir):
        torch.save(self.state_dict(),os.path.join(dir,'ConvAE.pt'))


    def load_model_(self,dir):
        self.load_state_dict(torch.load(os.path.join(dir,'ConvAE.pt')))