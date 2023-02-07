from torch.utils.data import DataLoader
from src.embed import embed_text
from src.pair_data import LIWCEmbedData, PairData
from src.model import Net, NetWithRoberta
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from src.utils import save, load
from src.test import test
from src.cluster import cluster
import os


def main():
    data_test = load("data_test")
    data_train = load("data_train")
    model = NetWithRoberta()
    # model.load_state_dict(load("net13023/net_8"))
    model.load_state_dict(load("net_18"))

    # cluster(model, data_test.liwc_data)
    print(test(model, data_test))
    print(test(model, data_train))


if __name__ == "__main__":
    main()
