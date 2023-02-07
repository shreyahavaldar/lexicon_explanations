from src.utils import save, load
from sklearn.cluster import AgglomerativeClustering
import numpy as np
import torch
from tqdm import tqdm


def cluster(model, data):
    model = model.cuda()
    model.eval()
    # for idx, ((word1, word2), _) in tqdm(enumerate(data)):
    #     i = data.idxs[idx][0]
    #     j = data.idxs[idx][1]
    #     with torch.no_grad():
    #         distances[i, j] = model(word1, word2)
    #         distances[j, i] = distances[i, j]

    distances = load("distance_matrix")
    if distances is None:
        distances = torch.zeros((len(data), len(data)))
        for i in tqdm(range(len(data))):
            for j in range(len(data)):
                if j <= i:
                    continue
                word1 = {k: v[None, :].cuda() for k, v in data[i][0].items()}
                word2 = {k: v[None, :].cuda() for k, v in data[j][0].items()}
                with torch.no_grad():
                    distances[i, j] = model(word1, word2).cpu()
                distances[j, i] = distances[i, j]

        distances = torch.sigmoid(distances).numpy()
        save(distances, "distance_matrix")

    clustering = AgglomerativeClustering(n_clusters=72).fit(distances).labels_
    for c in range(72):
        print("Words in cluster", c)
        for i in np.nonzero(clustering == c)[0]:
            print(data.data.iloc[i]["term"], data.data.iloc[i]["groups"])
        print()
