import os
import random

import librosa
import pandas as pd
import torch
import audioread

import logging
logger = logging.getLogger(__name__)

from utils import convert_audio


class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, config, transform=None,mode='train'):
        assert mode in ['train', 'test'], 'dataset mode must be train or test'
        if mode == 'train':
            self.audio_files = pd.read_csv(config.datasets.train_csv_path,on_bad_lines='skip')
        elif mode == 'test':
            self.audio_files = pd.read_csv(config.datasets.test_csv_path,on_bad_lines='skip',)
        self.transform = transform
        self.fixed_length = config.datasets.fixed_length
        self.tensor_cut = config.datasets.tensor_cut
        self.sample_rate = config.model.sample_rate
        self.channels = config.model.channels

    def __len__(self):
        return self.fixed_length if self.fixed_length and len(self.audio_files) > self.fixed_length else len(self.audio_files)  

    def get(self, idx=None):
        """uncropped, untransformed getter with random sample feature"""
        if idx is not None and idx > len(self.audio_files):
            raise StopIteration
        if idx is None:
            idx = random.randrange(len(self))
        try:
            logger.debug(f'Loading {self.audio_files.iloc[idx, :].values[0]}')
            waveform, sample_rate = librosa.load(
                self.audio_files.iloc[idx, :].values[0], 
                sr=self.sample_rate,
                mono=self.channels == 1
            )
        except (audioread.exceptions.NoBackendError, ZeroDivisionError):
            logger.warning(f"Not able to load {self.audio_files.iloc[idx, :].values[0]}, removing from dataset")
            self.audio_files.drop(idx, inplace=True)
            return self[idx]

        # add channel dimension IF loaded audio was mono
        waveform = torch.as_tensor(waveform)
        if len(waveform.shape) == 1:
            waveform = waveform.unsqueeze(0)
            waveform = waveform.expand(self.channels, -1)

        return waveform, sample_rate

    def __getitem__(self, idx):
        # waveform, sample_rate = torchaudio.load(self.audio_files.iloc[idx, :].values[0])
        # """you can preprocess the waveform's sample rate to save time and memory"""
        # if sample_rate != self.sample_rate:
        #     waveform = convert_audio(waveform, sample_rate, self.sample_rate, self.channels)
        waveform, sample_rate = self.get(idx)

        if self.transform:
            waveform = self.transform(waveform)

        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1]-self.tensor_cut-1) # random start point
                waveform = waveform[:, start:start+self.tensor_cut] # cut tensor
                return waveform, sample_rate
            else:
                return waveform, sample_rate


def pad_sequence(batch):
    # Make all tensor in a batch the same length by padding with zeros
    batch = [item.permute(1, 0) for item in batch]
    batch = torch.nn.utils.rnn.pad_sequence(batch, batch_first=True, padding_value=0.)
    batch = batch.permute(0, 2, 1)
    return batch


def collate_fn(batch):
    tensors = []

    for waveform, _ in batch:
        tensors += [waveform]

    # Group the list of tensors into a batched tensor
    tensors = pad_sequence(tensors)
    return tensors