from torch.utils.data import DataLoader
from src.embed import embed_text
from src.pair_data import LIWCData, PairData
from src.model import Net
import torch.optim as optim
import torch.nn as nn
import torch
from tqdm import tqdm
from src.utils import save, load
import os

# output = embed_text("test")
# print(output.shape)


def main():
    data_train = load("data_train")
    if not data_train:
        data_train = PairData(LIWCData(
            "data/LIWC2015_processed.csv",
            split="train"))
        save(data_train, "data_train")

    data_test = load("data_test")
    if not data_test:
        data_test = PairData(
            LIWCData(
                "data/LIWC2015_processed.csv",
                split="test"))
        save(data_test, "data_test")

    train_dataloader = DataLoader(
        data_train,
        batch_size=40000,
        shuffle=True,
        num_workers=16)
    # test_dataloader = DataLoader(data_test, batch_size=64, shuffle=True)

    net = Net().cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.Adam(net.parameters(), lr=1e-2)

    for epoch in range(100):
        for i, data in enumerate(tqdm(train_dataloader)):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.cuda())
            loss = criterion(outputs, labels[:, None].float().cuda())
            loss.backward()
            optimizer.step()

            # if i % 200 == 199:
            if i % 10 == 0:
                tqdm.write(f"epoch: {epoch}, loss: {loss.item()}")
        save(net.state_dict(), f"net_{epoch}")


if __name__ == "__main__":
    main()
