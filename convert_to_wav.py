import os
from argparse import ArgumentParser

from pydub import AudioSegment


parser = ArgumentParser()
parser.add_argument('--aud')
args = parser.parse_args()

f = args.aud

ext = f.split('.')[-1]

aud = AudioSegment.from_file(f, ext)
aud.export(f.replace(ext,'wav'),format='wav')
os.remove(f)