import torch
import torch.nn as nn
import torch.nn.functional as F


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(1024, 512)
        self.fc1_share = nn.Linear(1024, 512)
        self.fc2 = nn.Linear(512, 128)
        self.fc3 = nn.Linear(128, 1)

    def forward(self, x):
        x1 = F.relu(self.fc1(x[:, :1024]))
        x2 = F.relu(self.fc1(x[:, 1024:]))
        x1_share = F.relu(self.fc1_share(x[:, :1024]))
        x2_share = F.relu(self.fc1_share(x[:, 1024:]))
        x1 = x1 + x2_share
        x2 = x2 + x1_share
        x1 = F.relu(self.fc2(x1))
        x2 = F.relu(self.fc2(x2))
        x = x1 + x2

        x = self.fc3(x)
        return x

    def embed(self, x):
        x1 = F.relu(self.fc1(x[:, :1024]))
        x2 = F.relu(self.fc1(x[:, 1024:]))
        x1_share = F.relu(self.fc1_share(x[:, :1024]))
        x2_share = F.relu(self.fc1_share(x[:, 1024:]))
        x1 = x1 + x2_share
        x2 = x2 + x1_share
        x1 = F.relu(self.fc2(x1))
        x2 = F.relu(self.fc2(x2))
        x = x1 + x2
        return x
