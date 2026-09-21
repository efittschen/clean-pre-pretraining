import torch
import numpy as np
from torch.utils.data import Dataset
from random import randrange
import os

#Dataset loader for Gutenberg structure-split data. Accepts individual train, test and dev jsonl files
#Adapted from https://github.com/timinar/BabyLlama/blob/main/babylm_dataset.py


def load_token_file(path):
    """Load a tokenized file as a memory-mapped numpy array wrapped in a torch tensor."""
    if os.path.exists(path):
        data = np.memmap(path, dtype=np.int64, mode="r")
        return torch.from_numpy(data)
    return torch.load(path)


class GBDataset(Dataset):

    def __init__(self, data_file, seq_length, offset=0, random_chunk=False, bos_token_id=None):
        self.seq_length = seq_length
        self.offset = offset
        self.random_chunk = random_chunk
        self.bos_token_id = bos_token_id
        self.content_length = seq_length - 1 if bos_token_id is not None else seq_length
        if isinstance(data_file, str):
            self.data = load_token_file(data_file)
        else:
            self.data = data_file

    def __len__(self):
        if self.random_chunk:
            return max(0, len(self.data) // self.content_length - 1)
        else:
            return max(0, (len(self.data) - self.offset) // self.content_length)

    def __getitem__(self, i):
        if self.random_chunk:
            offset = randrange(self.content_length)
            chunk = self.data[i*self.content_length+offset:(i+1)*self.content_length+offset]
        else:
            chunk = self.data[i*self.content_length+self.offset:(i+1)*self.content_length+self.offset]
        if self.bos_token_id is not None:
            chunk = torch.cat([torch.tensor([self.bos_token_id]), chunk])
        return chunk
        
    
