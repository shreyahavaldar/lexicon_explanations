from torch.utils.data import DataLoader
from src.embed import embed_text
from src.pair_data import LIWCTokenData, LIWCEmbedData, PairData
from src.model import Net, PretrainedBERT
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from src.utils import save, load
import os
import numpy as np
import copy
from torch.utils.data.sampler import WeightedRandomSampler


def get_all_labels(dataloader):
    labels = []
    for (_, _), label in tqdm(dataloader):
        label_cp = copy.deepcopy(label)
        del label
        labels.append(label_cp)
    return torch.concat(labels)


class CustomWeightedRandomSampler(WeightedRandomSampler):
    """WeightedRandomSampler except allows for more than 2^24 samples to be sampled"""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def __iter__(self):
        rand_tensor = np.random.choice(range(0, len(self.weights)),
                                       size=self.num_samples,
                                       p=self.weights.numpy() / torch.sum(self.weights).numpy(),
                                       replace=self.replacement)
        rand_tensor = torch.from_numpy(rand_tensor)
        return iter(rand_tensor.tolist())


def main():
    data_train = load("data_train")
    if data_train is None:
        data_train = PairData(LIWCEmbedData(
            "data/LIWC2015_processed.csv",
            split="train"))
        save(data_train, "data_train")

    data_test = load("data_test")
    if data_test is None:
        data_test = PairData(
            LIWCEmbedData(
                "data/LIWC2015_processed.csv",
                split="val"))
        save(data_test, "data_test")

    train_dataloader = DataLoader(
        data_train,
        batch_size=20000,
        shuffle=False,
        num_workers=32)

    print(data_train[0])
    print(data_train[0][0][0].shape)

    y = load("y_train")
    if y is None:
        y = get_all_labels(train_dataloader)
        save(y, "y_train")

    counts = np.array([torch.sum(y == 0), torch.sum(y == 1)])
    labels_weights = 1. / counts
    weights = labels_weights[y]
    sampler = CustomWeightedRandomSampler(weights, len(weights))

    train_dataloader = DataLoader(
        data_train,
        batch_size=20000,
        # shuffle=True,
        sampler=sampler,
        num_workers=16)
    # test_dataloader = DataLoader(data_test, batch_size=64, shuffle=True)

    net = Net().cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(net.parameters(), lr=1e-3)

    for epoch in range(100):
        for i, data in enumerate(tqdm(train_dataloader)):
            (inputs1, inputs2), labels = data
            inputs1 = inputs1.cuda()
            inputs2 = inputs2.cuda()
            # print("Fraction of 1s:", torch.sum(labels) / labels.shape[0])
            optimizer.zero_grad()
            outputs = net(inputs1, inputs2)
            loss = criterion(outputs, labels[:, None].float().cuda())
            loss.backward()
            optimizer.step()

            # if i % 200 == 199:
            if i % 10 == 0:
                tqdm.write(f"epoch: {epoch}, loss: {loss.item()}")
        save(net.state_dict(), f"net_{epoch}")


if __name__ == "__main__":
    main()
