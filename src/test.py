import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, mean_squared_error
import numpy as np
import copy
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def test(model, data):
    test_dataloader = DataLoader(
        data,
        batch_size=5000,
        shuffle=False,
        num_workers=16)

    model = model.cuda()
    model.eval()
    preds = []
    gt = []
    for i, data in enumerate(tqdm(test_dataloader)):
        inputs, labels = data
        label_cp = copy.deepcopy(labels)
        del labels
        with torch.no_grad():
            outputs = model(inputs.cuda()).cpu()
        preds.append(torch.round(torch.sigmoid(outputs)))
        gt.append(label_cp)
    preds = torch.concat(preds, dim=0).numpy()
    gt = torch.concat(gt, dim=0).numpy()
    print(preds)
    print(gt)

    metrics = {
        "accuracy": accuracy_score(gt, preds),
        "recall": recall_score(gt, preds),
        "inverse-recall": recall_score(1 - gt, 1 - preds),
        "precision": precision_score(gt, preds),
        "mse": mean_squared_error(gt, preds),
        "baseline": accuracy_score(gt, np.zeros(gt.shape))}
    return metrics
