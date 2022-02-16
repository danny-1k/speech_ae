import os
import numpy as np


import argparse


from tqdm import tqdm


parser = argparse.ArgumentParser()
parser.add_argument('--dataset',required=True)

args = parser.parse_args()

dataset = args.dataset


if dataset not in ['mel_spec','mfcc','waveform']:
    raise ValueError('`dataset` must be mel_spec,mfcc or waveform')

DATA_PATH = f'../data/processed/{dataset}'

files = os.listdir(DATA_PATH)

print('CALCULATING STATS FOR THE WHOLE DATASET')
print('VALUES USED HERE SHOULD NOT BE USED TO PREPROCESS THE MODEL')
print('USE `calculate_train_stats.py` TO CALCULATE FOR THE TRAIN DATA')
print('WHICH CAN BE USED FOR PREPROCESSING OF TRAIN & TEST')
print()
print()
print('THIS WILL TAKE A PRETTY LONG TIME')
print('SO FEEL FREE TO GRAB A CUP OF COFFEE')
print()

MEAN = None
STD = None
MAX = None
MIN = None


for f in tqdm(files):
    arr = np.load(os.path.join(DATA_PATH,f))

    if isinstance(MEAN,type(None)):
        MEAN = arr.mean()

    else:
        MEAN += arr.mean()


    if isinstance(STD,type(None)):
        STD = arr.std()

    else:
        STD += arr.std()


    if isinstance(MAX,type(None)):
        MAX = arr.max()

    else:
        MAX += arr.max()

    
    if isinstance(MIN,type(None)):
        MIN = arr.min()
    else:
        MIN += arr.min()




MEAN /= len(files)
STD /= len(files)
MAX /= len(files)
MIN /= len(files)

print(f'MEAN (ALL DIMS) => {MEAN}')
print(f'STD  (ALL DIMS) => {STD}')
print(f'MAX             => {MAX}')
print(f'MIN             => {MIN}')