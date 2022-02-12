import os
import shutil
import random

from tqdm import tqdm


from config import MelSpecConfig

files = [os.path.join('../data/processed/mel_spec',f) for f in os.listdir('../data/processed/mel_spec')]

random.shuffle(files)

print('TRAIN_SIZE => ',int(len(files)*MelSpecConfig.TRAIN_PCT))
print('TEST_SIZE => ',len(files)-int(len(files)*MelSpecConfig.TRAIN_PCT))

train = files[:int(len(files)*MelSpecConfig.TRAIN_PCT)]
test = files[int(len(files)*MelSpecConfig.TRAIN_PCT):]



for t in tqdm(train):
    if t.replace('\\','/').split('/')[-1] in os.listdir('../data/train/mel_spec'):
        continue
    else:shutil.copy(t,os.path.join('../data/train/mel_spec',t.replace('\\','/').split('/')[-1]))


for t in tqdm(test):
    if t.replace('\\','/').split('/')[-1] in os.listdir('../data/test/mel_spec'):
        continue
    else:shutil.copy(t,os.path.join('../data/test/mel_spec',t.replace('\\','/').split('/')[-1]))