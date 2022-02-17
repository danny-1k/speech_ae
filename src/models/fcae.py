import os

import torch
import torch.nn as nn


class FCAE(nn.Module):
    def __init__(self):
        super().__init__()

        self.encoder = nn.Sequential(

            nn.Linear(20*5,50),
            nn.Linear(50,50),
        )

        self.decoder = nn.Sequential(
            nn.Linear(50,50),
            nn.Linear(50,20*5),

            
        )


    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        return x


    def save_model(self,dir):
        torch.save(self.state_dict(),os.path.join(dir,'FCAE.pt'))


    def load_model_(self,dir):
        self.load_state_dict(torch.load(os.path.join(dir,'FCAE.pt')))