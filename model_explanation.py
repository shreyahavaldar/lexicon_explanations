import torch
import numpy as np
import scipy as sp
import shap
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.pair_data import LIWCWordData
from src.utils import *
from src.train import train
from tqdm import tqdm
import pandas as pd
import random


def main():
    config = {"dataset": "yelp", "topics": "neurallda"}
    model1, model2 = load_models(config)

    data_train, data_val, data_test = load_data(config)
    print(len(data_train), len(data_val), len(data_test))

    x = ([data_train[i]['sentence'] for i in range(len(data_train))]
        + [data_val[i]['sentence'] for i in range(len(data_val))]
        + [data_test[i]['sentence'] for i in range(len(data_test))])
    x = [xi for xi in x if len(xi.split()) > 1]

    print("Num docs for topic model:", len(x))
    topics, word2idx = get_topics(config, x)

    train(config, model1, data_train, data_val)

    # Only evaluate models on the test data
    x = [data_test[i]['sentence'] for i in range(len(data_test))]
    topic_vals, word_vals = get_topic_shap(model1, x, topics, word2idx)
    save(topic_vals, f"topic_vals_distilroberta_{config['dataset']}")
    save(word_vals, f"word_vals_distilroberta_{config['dataset']}")
    del model1.model
    del model1
    torch.cuda.empty_cache()

    train(config, model2, data_train, data_val)
    topic_vals, word_vals = get_topic_shap(model2, x, topics, word2idx)
    save(topic_vals, f"topic_vals_gpt2_{config['dataset']}")
    save(word_vals, f"word_vals_gpt2_{config['dataset']}")


if __name__ == "__main__":
    main()
