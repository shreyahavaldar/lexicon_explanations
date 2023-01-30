import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from src.embed import embed_text, tokenize_text
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import itertools


class LIWCEmbedData(Dataset):
    def __init__(self, data_csv, split="train"):
        liwc_df = pd.read_csv(data_csv)
        liwc_df = liwc_df.groupby(
            by=["term"])["category"].apply(set).reset_index(
            name='groups')
        liwc_df_train, liwc_df_test = train_test_split(
            liwc_df, test_size=0.2, random_state=42)
        liwc_df_val, liwc_df_test = train_test_split(
            liwc_df_test, test_size=0.5, random_state=42)

        if split == "train":
            liwc_df = liwc_df_train
        elif split == "val":
            liwc_df = liwc_df_val
        elif split == "test":
            liwc_df = liwc_df_test

        self.data = liwc_df
        self.len = len(liwc_df)

        tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        model = AutoModel.from_pretrained("roberta-large").cuda()

        embeddings = []
        for i in tqdm(range(self.len)):
            embeddings.append(
                embed_text(
                    model,
                    tokenizer,
                    self.data.iloc[i]["term"]).cpu())
        self.embeddings = torch.stack(embeddings)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        word = self.data.iloc[idx]
        emb = self.embeddings[idx, :]
        # print(word["term"])
        return emb, word["groups"]


class LIWCTokenData(Dataset):
    def __init__(self, data_csv, split="train"):
        liwc_df = pd.read_csv(data_csv)
        liwc_df = liwc_df.groupby(
            by=["term"])["category"].apply(set).reset_index(
            name='groups')
        liwc_df_train, liwc_df_test = train_test_split(
            liwc_df, test_size=0.2, random_state=42)
        liwc_df_val, liwc_df_test = train_test_split(
            liwc_df_test, test_size=0.5, random_state=42)

        if split == "train":
            liwc_df = liwc_df_train
        elif split == "val":
            liwc_df = liwc_df_val
        elif split == "test":
            liwc_df = liwc_df_test

        self.data = liwc_df
        self.len = len(liwc_df)

        tokenizer = AutoTokenizer.from_pretrained("roberta-large")

        self.tokens = tokenize_text(
            tokenizer, list(
                self.data["term"].to_list()))

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        word = self.data.iloc[idx]
        toks = {k: v[idx] for k, v in self.tokens.items()}
        return toks, word["groups"]


class PairData(Dataset):
    def __init__(self, liwc_data):
        self.liwc_data = liwc_data
        self.orig_len = len(self.liwc_data)
        self.idxs = list(itertools.combinations(list(range(self.orig_len)), r=2))

    def __len__(self):
        return len(self.idxs)

    def __getitem__(self, idx):
        i = self.idxs[idx][0]
        j = self.idxs[idx][1]
        # print(i, j)
        word1, groups1 = self.liwc_data[i]
        word2, groups2 = self.liwc_data[j]

        label = 1
        if len(groups1.intersection(groups2)) > 0:
            label = 0
        # concat_embed = torch.concat([word1, word2], dim=1).flatten()
        return (word1.flatten(), word2.flatten()), label
