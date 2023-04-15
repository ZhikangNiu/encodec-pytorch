import os
import pandas as pd
import torch
import torchaudio
import random


class CustomAudioDataset(torch.utils.data.Dataset):
    def __init__(self, csv_file, transform=None, tensor_cut=0, fixed_length=None):
        self.audio_labels = pd.read_csv(csv_file)
        self.transform = transform
        self.fixed_length = fixed_length
        self.tensor_cut = tensor_cut

    def __len__(self):
        if self.fixed_length:
            return self.fixed_length
        return len(self.audio_labels)

    def __getitem__(self, idx):
        waveform, sample_rate = torchaudio.load(self.audio_labels.iloc[idx, :].values[0])
        if self.transform:
            waveform = self.transform(waveform)

        if self.tensor_cut > 0:
            if waveform.size()[1] > self.tensor_cut:
                start = random.randint(0, waveform.size()[1]-self.tensor_cut-1)
                waveform = waveform[:, start:start+self.tensor_cut]
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