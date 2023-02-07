import torch
from tqdm import tqdm
from torch.utils.data import DataLoader
from sklearn.metrics import balanced_accuracy_score, f1_score, accuracy_score, precision_score, recall_score, mean_squared_error
import numpy as np
import copy
import torch.multiprocessing
torch.multiprocessing.set_sharing_strategy('file_system')


def test(model, data):
    test_dataloader = DataLoader(
        data,
        batch_size=10000,
        shuffle=False,
        num_workers=16)

    model = model.cuda()
    model.eval()
    preds = []
    gt = []
    for i, data in enumerate(tqdm(test_dataloader)):
        (word1, word2), labels = data
        word1 = {k: v.cuda() for k, v in word1.items()}
        word2 = {k: v.cuda() for k, v in word2.items()}
        label_cp = copy.deepcopy(labels)
        del labels
        with torch.no_grad():
            outputs = model(word1, word2).cpu()
        preds.append(torch.round(torch.sigmoid(outputs)))
        gt.append(label_cp)
    preds = torch.concat(preds, dim=0).numpy()
    gt = torch.concat(gt, dim=0).numpy()
    print(preds)
    print(gt)

    metrics = {
        "accuracy": balanced_accuracy_score(gt, preds),
        "f1-score": f1_score(1 - gt, 1 - preds),
        "recall": recall_score(gt, preds),
        "inverse-recall": recall_score(1 - gt, 1 - preds),
        "precision": precision_score(gt, preds),
        "mse": mean_squared_error(gt, preds),
        "baseline": accuracy_score(gt, np.zeros(gt.shape))}
    return metrics
