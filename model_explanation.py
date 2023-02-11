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


def main():
    config = {"dataset": "cola", "topics": "liwc"}
    model1, model2 = load_models(config)

    data_train, data_val = load_data(config)
    x = [data_val[i]['sentence'] for i in range(len(data_val))]
    x = [xi for xi in x if len(xi.split()) > 1]

    topics, topic_names, word2idx = get_topics(config, data_train)

    train(config, model1, data_train, data_val)
    topic_vals, word_vals = get_topic_shap(model1, x, topics, word2idx)
    save(topic_vals, f"topic_vals_roberta_{config['dataset']}")
    save(word_vals, f"word_vals_roberta_{config['dataset']}")
    del model1.model
    del model1
    torch.cuda.empty_cache()

    train(config, model2, data_train, data_val)
    topic_vals, word_vals = get_topic_shap(model2, x, topics, word2idx)
    save(topic_vals, f"topic_vals_gpt2_{config['dataset']}")
    save(word_vals, f"word_vals_gpt2_{config['dataset']}")


if __name__ == "__main__":
    main()
