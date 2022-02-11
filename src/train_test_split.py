import os
import shutil
import random

from tqdm import tqdm


TRAIN_PCT = .8

files = [os.path.join('../data/processed',f) for f in os.listdir('../data/processed')]

random.shuffle(files)

print('TRAIN_SIZE => ',int(len(files)*TRAIN_PCT))
print('TEST_SIZE => ',len(files)-int(len(files)*TRAIN_PCT))

train = files[:int(len(files)*TRAIN_PCT)]
test = files[int(len(files)*TRAIN_PCT):]



for t in tqdm(train):
    if t.replace('\\','/').split('/')[-1] in os.listdir('../data/train'):
        continue
    else:shutil.copy(t,os.path.join('../data/train',t.replace('\\','/').split('/')[-1]))


for t in tqdm(test):
    if t.replace('\\','/').split('/')[-1] in os.listdir('../data/test'):
        continue
    else:shutil.copy(t,os.path.join('../data/test',t.replace('\\','/').split('/')[-1]))