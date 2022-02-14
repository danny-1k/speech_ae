import os
import shutil
import random

from tqdm import tqdm

from config import MelSpecConfig

import argparse


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',required=True)
parser.add_argument('--train_pct',required=True,type=float)


args = parser.parse_args()

dataset = args.dataset
train_pct = args.train_pct

if dataset not in ['mel_spec','mfcc','waveform']:
    raise ValueError('`dataset` must be mel_spec,mfcc or waveform')


if not (train_pct > 0 and train_pct < 1):
    raise ValueError('`train_pct` must be between 0 and 1')


files = [os.path.join(f'../data/processed/{dataset}',f) for f in os.listdir(f'../data/processed/{dataset}')]

random.shuffle(files)


if not os.path.exists(f'../data/train/{dataset}'):
    print(f'[!] {dataset} path in `train`does not exist... Creating')
    os.makedirs(f'../data/train/{dataset}')


if not os.path.exists(f'../data/test/{dataset}'):
    print(f'[!] {dataset} path in `test`does not exist... Creating')
    os.makedirs(f'../data/test/{dataset}')



if len(os.listdir(f'../data/train/{dataset}')) >0:
    print('[!] There are files in the `train` folder... Deleting')
    for f in tqdm(os.listdir(f'../data/train/{dataset}')):
        os.remove(os.path.join(f'../data/train/{dataset}',f))



if len(os.listdir(f'../data/test/{dataset}')) > 0:
    print('[!] There are files in the `test` folder... Deleting')
    for f in tqdm(os.listdir(f'../data/test/{dataset}')):
        os.remove(os.path.join(f'../data/test/{dataset}',f))


print('TRAIN_SIZE => ',int(len(files)*train_pct))
print('TEST_SIZE => ',len(files)-int(len(files)*train_pct))

train = files[:int(len(files)*train_pct)]
test = files[int(len(files)*train_pct):]



for t in tqdm(train):
    if t.replace('\\','/').split('/')[-1] in os.listdir(f'../data/train/{dataset}'):
        continue
    else:shutil.copy(t,os.path.join(f'../data/train/{dataset}',t.replace('\\','/').split('/')[-1]))


for t in tqdm(test):
    if t.replace('\\','/').split('/')[-1] in os.listdir(f'../data/test/{dataset}'):
        continue
    else:shutil.copy(t,os.path.join(f'../data/test/{dataset}',t.replace('\\','/').split('/')[-1]))