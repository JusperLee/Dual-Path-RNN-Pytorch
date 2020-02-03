import sys
sys.path.append('../')

from data_loader.AudioData import AudioReader
import torch
from torch.utils.data import Dataset

import numpy as np


class Datasets(Dataset):
    '''
       Load audio data
       mix_scp: file path of mix audio (type: str)
       ref_scp: file path of ground truth audio (type: list[spk1,spk2])
       chunk_size (int, optional): split audio size (default: 32000(4 s))
       least_size (int, optional): Minimum split size (default: 16000(2 s))
    '''

    def __init__(self, mix_scp=None, ref_scp=None, sample_rate=8000, chunk_size=32000, least_size=16000):
        super(Datasets, self).__init__()
        self.mix_audio = AudioReader(
            mix_scp, sample_rate=sample_rate, chunk_size=chunk_size, least_size=least_size).audio
        self.ref_audio = [AudioReader(
            r, sample_rate=sample_rate, chunk_size=chunk_size, least_size=least_size).audio for r in ref_scp]

    def __len__(self):
        return len(self.mix_audio)

    def __getitem__(self, index):
        return self.mix_audio[index], [ref[index] for ref in self.ref_audio]


if __name__ == "__main__":
    dataset = Datasets("/home/likai/data1/create_scp/cv_mix.scp",
                      ["/home/likai/data1/create_scp/cv_s1.scp", "/home/likai/data1/create_scp/cv_s2.scp"])
    for i in dataset.mix_audio:
        if i.shape[0] != 32000:
            print('fail')
