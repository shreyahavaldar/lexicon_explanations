import torch
from torch.utils.data import Dataset
from transformers import AutoTokenizer, AutoModel
from src.embed import embed_text
import pandas as pd
from sklearn.model_selection import train_test_split
from tqdm import tqdm


class LIWCData(Dataset):
    def __init__(self, data_csv, split="train"):
        liwc_df = pd.read_csv(data_csv)
        liwc_df_train, liwc_df_test = train_test_split(
            liwc_df, test_size=0.2, random_state=42)
        liwc_df = liwc_df_train if split == "train" else liwc_df_test
        liwc_df = liwc_df.groupby(
            by=["term"])["category"].apply(set).reset_index(
            name='groups')
        self.data = liwc_df
        self.len = len(liwc_df)

        self.tokenizer = AutoTokenizer.from_pretrained("roberta-large")
        self.model = AutoModel.from_pretrained("roberta-large").cuda()

        embeddings = []
        for i in tqdm(range(self.len)):
            embeddings.append(
                embed_text(
                    self.model,
                    self.tokenizer,
                    self.data.iloc[i]["term"]))
        self.embeddings = torch.stack(embeddings)

    def __len__(self):
        return self.len

    def __getitem__(self, idx):
        word = self.data.iloc[idx]
        emb = self.embeddings[idx, :]
        # print(word["term"])
        return emb, word["groups"]


class PairData(Dataset):
    def __init__(self, liwc_data):
        self.liwc_data = liwc_data
        self.orig_len = len(self.liwc_data)

    def __len__(self):
        return self.orig_len * self.orig_len

    def __getitem__(self, idx):
        i = idx // self.orig_len
        j = idx % self.orig_len
        # print(i, j)
        word1, groups1 = self.liwc_data[i]
        word2, groups2 = self.liwc_data[j]

        label = 1
        if len(groups1.intersection(groups2)) > 0:
            label = 0
        concat_embed = torch.concat([word1, word2], dim=0)
        return concat_embed, label
