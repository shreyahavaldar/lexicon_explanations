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
        batch_size=256,
        shuffle=True,
        num_workers=0)
    # test_dataloader = DataLoader(data_test, batch_size=64, shuffle=True)

    net = Net().cuda()

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.SGD(net.parameters(), lr=0.001, momentum=0.9)

    # for i in tqdm(range(len(data_train))):
    #     data, label = data_train[i]
    # print("finished test")

    # print(data_train[0])
    # print(data_train[1])

    for epoch in range(1):
        for i, data in enumerate(tqdm(train_dataloader)):
            inputs, labels = data
            optimizer.zero_grad()
            outputs = net(inputs.cuda())
            loss = criterion(outputs, labels[:, None].float().cuda())
            loss.backward()
            optimizer.step()

            # if i % 200 == 199:
            print(f"epoch: {epoch}, loss: {loss.item()}")
            if i % 100 == 99:
                save(net.state_dict(), "net")


if __name__ == "__main__":
    main()
