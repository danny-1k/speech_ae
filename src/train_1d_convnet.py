import torch
import torch.nn as nn
import torch.optim as optim

from models.waveae import WaveConv1D

from data import Waveform

from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau

from tqdm import tqdm

from utils.train_utils import save_loss_plot

MU = -0.00015
STD = 0.05

torch.manual_seed(42)

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train = Waveform(train=True,MU=MU,STD=STD)

test = Waveform(train=False,MU=MU,STD=STD)

trainloader = DataLoader(train,128,shuffle=True)
testloader = DataLoader(test,128,shuffle=True)

net = WaveConv1D()

net.to(device)

lossfn = nn.L1Loss()

lr = 3e-4

optimizer = optim.Adam(net.parameters(),lr=lr)

lrscheduler = ReduceLROnPlateau(optimizer,mode='min',patience=2,)

train_loss_over_time = []
test_loss_over_time = []


best_loss = float('inf')

for epoch in tqdm(range(300)):

    train_loss_epoch = []
    test_loss_epoch = []
    net.train()
    for x,y in trainloader:

        x = x.view(x.shape[0],-1)
        y = y.view(y.shape[0],-1)

        x = x.to(device)
        y = y.to(device)

        optimizer.zero_grad()

        p = net(x)
        loss = lossfn(p,y)
                
        loss.backward()

        optimizer.step()

        train_loss_epoch.append(loss.item())

            

    net.eval()
    with torch.no_grad():
        for x,y in testloader:

            x = x.view(x.shape[0],-1)
            y = y.view(y.shape[0],-1)

            x = x.to(device)
            y = y.to(device)

            p = net(x)
            loss = lossfn(p,y)
            test_loss_epoch.append(loss.item())

        
    train_loss_over_time.append(sum(train_loss_epoch)/len(train_loss_epoch))
    test_loss_over_time.append(sum(test_loss_epoch)/len(test_loss_epoch))


    lrscheduler.step(test_loss_over_time[-1])

    save_loss_plot(train_loss_over_time,
                    test_loss_over_time,
                    '../plots/WaveConv1D'
            )


    if test_loss_over_time[-1] <best_loss:
        best_loss = test_loss_over_time[-1]

        net.save_model('../models')