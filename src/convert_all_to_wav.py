import os
from argparse import ArgumentParser

from pydub import AudioSegment
from utils.utilities import list_dirs

from tqdm import tqdm


parser = ArgumentParser()
parser.add_argument('--audio_ds',action='store_true',default=False)
args = parser.parse_args()

if args.audio_ds:
    for f in tqdm(list_dirs('../data/audio_ds')):
        ext = f.split('.')[-1]
        if ext == 'wav':
            continue
        aud = AudioSegment.from_file(f, ext)
        aud.export(f.replace(ext,'wav'),format='wav')
        os.remove(f)

else:
    for f in tqdm(os.listdir('../data/processed')):
        path = os.path.join('../data/processed',f)
        ext = path.split('.')[-1]
        if ext == 'wav':
            continue
        aud = AudioSegment.from_file(path, ext)
        aud.export(path.replace(ext,'wav'),format='wav')
        os.remove(path)