import torch
from torch.utils.data import Dataset
import pandas as pd


class LIWCData(Dataset):
    def __init__(self, data_csv):
        liwc_df = pd.read_csv(data_csv)
        liwc_df = liwc_df.groupby(
            by=["term"])["category"].apply(set).reset_index(
            name='groups')
        self.data = liwc_df
        self.len = len(liwc_df)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        word = self.data.iloc[idx]
        return word["term"], word["groups"]


class PairData(Dataset):
    def __init__(self, liwc_data):
        self.liwc_data = liwc_data
        self.orig_len = len(self.liwc_data)

    def __len__(self):
        return self.orig_len * (self.orig_len - 1)

    def __getitem__(self, idx):
        i = idx // self.orig_len
        j = idx % self.orig_len
        word1, groups1 = self.liwc_data[i]
        word2, groups2 = self.liwc_data[j]

        label = 1
        if len(groups1.intersection(groups2)) > 0:
            label = 0
        return (word1, word2), label
