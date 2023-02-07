import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(384, 384)
        self.fc1_share = nn.Linear(384, 384)
        self.fc2 = nn.Linear(384, 1)
        # self.fc3 = nn.Linear(128, 1)

    def forward(self, x1, x2):
        x1_a = self.fc1(x1)
        x2_a = self.fc1(x2)
        x1_share = self.fc1_share(x1)
        x2_share = self.fc1_share(x2)
        x1 = F.relu(x1_a + x2_share)
        x2 = F.relu(x2_a + x1_share)
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        x = x1 + x2
        return x

    def embed(self, x):
        x1 = F.relu(self.fc1(x[:, :384]))
        x2 = F.relu(self.fc1(x[:, 384:]))
        x1_share = F.relu(self.fc1_share(x[:, :384]))
        x2_share = F.relu(self.fc1_share(x[:, 384:]))
        x1 = x1 + x2_share
        x2 = x2 + x1_share
        x1 = F.relu(self.fc2(x1))
        x2 = F.relu(self.fc2(x2))
        x = x1 + x2
        return x


class NetWithRoberta(nn.Module):

    def __init__(self):
        super(NetWithRoberta, self).__init__()
        self.roberta = PretrainedBERT()
        self.fc1 = nn.Linear(384, 64)
        self.fc1_share = nn.Linear(384, 64)
        self.fc2 = nn.Linear(64, 1)
        # self.fc3 = nn.Linear(128, 1)

    def forward(self, x1, x2):
        x1 = self.roberta(x1)
        x2 = self.roberta(x2)
        x1_a = self.fc1(x1)
        x2_a = self.fc1(x2)
        x1_share = self.fc1_share(x1)
        x2_share = self.fc1_share(x2)
        x1 = F.relu(x1_a + x2_share)
        x2 = F.relu(x2_a + x1_share)
        x1 = self.fc2(x1)
        x2 = self.fc2(x2)
        x = x1 + x2
        return x

    def embed(self, x):
        x1 = F.relu(self.fc1(x[:, :384]))
        x2 = F.relu(self.fc1(x[:, 384:]))
        x1_share = F.relu(self.fc1_share(x[:, :384]))
        x2_share = F.relu(self.fc1_share(x[:, 384:]))
        x1 = x1 + x2_share
        x2 = x2 + x1_share
        x1 = F.relu(self.fc2(x1))
        x2 = F.relu(self.fc2(x2))
        x = x1 + x2
        return x


class PretrainedBERT(nn.Module):
    def __init__(self):
        super(PretrainedBERT, self).__init__()
        self.model = AutoModel.from_pretrained(
            'sentence-transformers/all-MiniLM-L6-v2').cuda()

    def forward(self, x):
        output = torch.mean(self.model(**x)[0], dim=1)
        return F.normalize(output, p=2, dim=1)
