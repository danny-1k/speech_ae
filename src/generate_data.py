import os
import json
import argparse
from random import sample
import numpy as np
from tqdm import tqdm
import utils.audio_utils as audio_utils
from utils.utilities import list_dirs

from torchaudio.transforms import MelSpectrogram,AmplitudeToDB,MFCC


class DataGen:
    def __init__(self,dataset_type='mel_spec',params={},time_per_sample=1,ignore_warning=False):
        assert dataset_type in ['mel_spec','waveform','mfcc'], f'`dataset_type` not valid. Got {dataset_type}'
        self.dataset_type = dataset_type
        self.time_per_sample = time_per_sample
        self.sampling_rate = 16000 # speech
        self.idx = 0 # keep track of files
        self.save_dir = os.path.join('../data/processed',
                        'mel_spec' if dataset_type=='mel_spec' \
                                    else 'waveform' \
                                        if dataset_type=='waveform' \
                                            else 'mfcc'
                        )
        self.params = params
        self.ignore_warning = ignore_warning

        if dataset_type == 'mel_spec':
            self.transform = MelSpectrogram(
                sample_rate=self.sampling_rate,
                n_fft=self.params.get('n_fft'),
                hop_length=self.params.get('hop_length'),
                n_mels=self.params.get('n_mels'),

            )


        elif dataset_type == 'waveform':
            self.transform = lambda x:x

        
        elif dataset_type == 'mfcc':
            self.transform = MFCC(
                sample_rate=self.sampling_rate,
                n_mfcc=self.params.get('n_mfcc'),
                log_mels=self.params.get('log_mels'),
                melkwargs={'n_fft':1024,'hop_length':512,}

            ) 


        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)


        if len(os.listdir(self.save_dir)) !=0:
            print(f'[!] There are {len(os.listdir(self.save_dir))} files in the saving directory')


            if self.ignore_warning:
                self.clear_save_dir()

            else:


                if input('Continue [y/n]') in 'yY':
                    self.clear_save_dir()

                else:
                    raise RuntimeError('Stopped dataset generation')


    def clear_save_dir(self):
        print('[!] Removing files found')
        for f in os.listdir(self.save_dir):
            os.remove(os.path.join(self.save_dir,f))


    def resample(self,audio,sr):
        return audio_utils.resample(audio,sr,self.sampling_rate)


    def read_audio(self,f):
        audio, sr = audio_utils.read_audio_file(f)
        return audio,sr


    def convert_to_mono(self,audio):
        return audio_utils.convert_to_mono(audio)


    def adjust_time(self,audio):
        return audio_utils.cut_to_max_time(audio,self.sampling_rate,self.time_per_sample)


    def save(self,obj):
        np.save(os.path.join(self.save_dir,f'{self.idx}.npy'),obj)


    def make_dataset(self):
        number_of_samples = int(self.sampling_rate*self.time_per_sample)

        for f in tqdm(list_dirs('../data/audio_ds')):
            audio,sr = self.read_audio(f)
            audio = self.convert_to_mono(audio)
            audio = self.resample(audio,sr)


            number_of_splits = audio.shape[-1] // int(number_of_samples) #nothing should go to waste XD

            if len(audio.shape)>1:
                audio = audio[0]

            if number_of_splits < 1:

                audio = self.adjust_time(audio,self.sampling_rate,self.time_per_sample)
                self.transform(audio)
                audio = audio.numpy()
                self.save(audio)

                self.increment_idx()
            
            else:
                for split in range(number_of_splits):

                    slice_ = audio[split*number_of_samples:][:number_of_samples]
                    slice_ = self.transform(slice_)
                    slice_ = slice_.numpy()
                    self.save(slice_)

                    self.increment_idx()

    
    def increment_idx(self):
        self.idx += 1



if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument('--time_per_sample',required=True,type=float)
    parser.add_argument('--dataset',required=True,type=str)
    parser.add_argument('--params',required=False,type=str,default=None)
    parser.add_argument('--ignore_warning',action='store_true')

    args = parser.parse_args()

    time_per_sample = args.time_per_sample
    dataset = args.dataset

    if args.params != None:
        params = ''
        for idx, param in enumerate(args.params.replace(' ','').split(',')):
            key_val = param.split(':')
            key = f'"{key_val[0]}"'
            val = key_val[1]

            params+=':'.join([key,val]) + (',' if idx != args.params.count(',') else '')

        params = json.loads('{' + params + '}')

    else:
        params = args.params

    ignore_warning = args.ignore_warning

    gen = DataGen(params=params,
                dataset_type=dataset,
                time_per_sample=time_per_sample,
                ignore_warning=ignore_warning
        )


    gen.make_dataset()