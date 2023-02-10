import torch
import os


def save(obj, name):
    torch.save(obj, os.path.join("output", name + ".pt"))


def load(name):
    if os.path.isfile(os.path.join("output", name + ".pt")):
        return torch.load(os.path.join("output", name + ".pt"))
    elif os.path.isfile(os.path.join("../output", name + ".pt")):
        return torch.load(os.path.join("../output", name + ".pt"))
    else:
        return None
