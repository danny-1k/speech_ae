import torch
import torch.nn as nn
import torch.optim as optim

from models.convae import ConvAE

from data import SpeechDS

from torch.utils.data import DataLoader

from torch.optim.lr_scheduler import StepLR,ReduceLROnPlateau

from tqdm import tqdm

from utils.train_utils import save_loss_plot

torch.manual_seed(42)


device = 'cuda' if torch.cuda.is_available() else 'cpu'

train = SpeechDS(train=True)

test = SpeechDS(train=False)

trainloader = DataLoader(train,32,shuffle=True)
testloader = DataLoader(test,32,shuffle=True)

# net = WaveNet(int(train.sample_rate*train.time_per_sample))

# net.to(device)

net = ConvAE()

lossfn = nn.MSELoss()

lr = 1e-2

optimizer = optim.Adam(net.parameters(),lr=lr)

lrscheduler = ReduceLROnPlateau(optimizer,mode='min',patience=2,)

train_loss_over_time = []
test_loss_over_time = []


best_loss = float('inf')

for epoch in tqdm(range(100)):

    train_loss_epoch = []
    test_loss_epoch = []
    net.train()
    for x,y in trainloader:

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
                    '../plots/ConvAELoss'
            )


    if test_loss_over_time[-1] <best_loss:
        best_loss = test_loss_over_time[-1]

        net.save_model('../models')