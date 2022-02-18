import os

import torch
import torch.nn as nn



class WaveConv1D(nn.Module):
    def __init__(self,depth=2,width=32,starting=16):
        super().__init__()
        self.depth = depth
        self.width = width
        self.starting = starting

        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1,out_channels=16,
                        kernel_size=3,stride=2,padding=3//2
            ), #(16,1200)

            nn.LeakyReLU(),

            nn.Conv1d(in_channels=16,out_channels=32,
                        kernel_size=3,stride=2,padding=3//2
            ), #(32,600)


            nn.LeakyReLU(),

            nn.Conv1d(in_channels=32,out_channels=64,
                        kernel_size=3,stride=2,padding=3//2
            ), # (64,300)


            nn.LeakyReLU(),

            nn.Conv1d(in_channels=64,out_channels=64,
                        kernel_size=3,stride=2,padding=3//2
            ), # (64,150)


            nn.LeakyReLU(),

            nn.Conv1d(in_channels=64,out_channels=64,
                        kernel_size=3,stride=2,padding=3//2
            ), # (64, 75)


        )

        self.decoder = nn.Sequential(
            nn.ConvTranspose1d(in_channels=64,out_channels=64,
                        kernel_size=4,stride=2,padding=3//2
            ), # (64,150)

            nn.LeakyReLU(),

            nn.ConvTranspose1d(in_channels=64,out_channels=64,
                        kernel_size=4,stride=2,padding=3//2
            ), # (64,300)

            nn.LeakyReLU(),


            nn.ConvTranspose1d(in_channels=64,out_channels=32,
                        kernel_size=4,stride=2,padding=3//2
            ), # (64,600)

            nn.LeakyReLU(),


            nn.ConvTranspose1d(in_channels=32,out_channels=16,
                        kernel_size=4,stride=2,padding=3//2
            ), # (64,1200)

            nn.LeakyReLU(),


            nn.ConvTranspose1d(in_channels=16,out_channels=1,
                        kernel_size=4,stride=2,padding=3//2
            ), # (64,2400)

        )


    def forward(self,x):

        x = self.encoder(x)
        x = self.decoder(x)

        # print(x.shape)

        return x


    def save_model(self,dir):
        torch.save(self.state_dict(),os.path.join(dir,'WaveConv1D.pt'))


    def load_model_(self,dir):
        self.load_state_dict(torch.load(os.path.join(dir,'WaveConv1D.pt')))