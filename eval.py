from torch.utils.data import DataLoader
from src.embed import embed_text
from src.pair_data import LIWCEmbedData, PairData
from src.model import Net
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from src.utils import save, load
from src.test import test
import os


def main():
    data_test = load("data_test")
    data_train = load("data_train")
    model = Net()
    model.load_state_dict(load("net_1"))
    print(test(model, data_test))
    print(test(model, data_train))


if __name__ == "__main__":
    main()
