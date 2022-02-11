import os
import numpy as np


DATA_PATH = '../data/train'

files = os.listdir(DATA_PATH)

print('CALCULATING STATS FOR THE TRAIN SET')

print('THIS WILL TAKE A PRETTY LONG TIME')
print('SO FEEL FREE TO GRAB A CUP OF COFFEE')
print()

MEAN = None
STD = None
MAX = None
MIN = None


for f in files:
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