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
    config = {"dataset": "goemotions", "topics": "liwc"}
    model1, model2 = load_models(config)

    data_train, data_val, data_test = load_data(config)
    print(len(data_train), len(data_val), len(data_test))

    x = [data_train[i]['sentence'] for i in range(len(data_train))]
    if config["dataset"] != "blog" and config["dataset"] != "yelp":
        x += ([data_val[i]['sentence'] for i in range(len(data_val))]
        + [data_test[i]['sentence'] for i in range(len(data_test))])

    indices = list(range(len(x)))
    if config["dataset"] == "blog" or config["dataset"] == "yelp":
        random.seed(316)
        random.shuffle(indices)
        x = [x[docid][:55000] for docid in indices[:200000]]
        data_train = data_train.select(indices[:200000])
    else:
        x = [x[docid][:55000] for docid in indices]

    # x = [xi for xi in x if len(xi.split()) > 1]

    print("Num docs for topic model:", len(x))
    topics, word2idx = get_topics(config, x)

    train(config, model1, data_train, data_val, data_test, batch_size=16, lr=5e-5)

    # Only evaluate models on the test data with more than 1 token
    def tokenize_function(examples):
        return len(model2.tokenizer.tokenize(examples["sentence"], padding=False, truncation=True, max_length=512)) > 1

    #  Filter then select only for blog
    if config["dataset"] == "blog":
        data_test_shap = data_test.filter(tokenize_function)
        indices = list(range(len(data_test_shap)))
        random.seed(316)
        random.shuffle(indices)
        data_test_shap = data_test_shap.select(indices[:1000])
    else:
        indices = list(range(len(data_test)))
        random.seed(316)
        random.shuffle(indices)
        data_test_shap = data_test.select(indices[:1000])
        data_test_shap = data_test_shap.filter(tokenize_function)

    x = [data_test_shap[i]['sentence'] for i in range(len(data_test_shap))]

    shap_vals = load(f"shap_vals_distilroberta_{config['dataset']}")
    shap_vals, topic_vals, word_vals, topics, idx2stop_word_topic = get_topic_shap(model1, model2, x, topics, word2idx, shap_vals)
    save(topic_vals, f"topic_vals_distilroberta_{config['dataset']}_{config['topics']}")
    save(word_vals, f"word_vals_distilroberta_{config['dataset']}_{config['topics']}")
    save(shap_vals, f"shap_vals_distilroberta_{config['dataset']}")
    save(idx2stop_word_topic, f"stopword_topic_names_{config['dataset']}_{config['topics']}")
    # del model1.model
    # del model1
    # torch.cuda.empty_cache()

    train(config, model2, data_train, data_val, data_test, batch_size=8, lr=5e-5)
    shap_vals = load(f"shap_vals_gpt2_{config['dataset']}")
    shap_vals, topic_vals, word_vals, topics, _ = get_topic_shap(model2, model1, x, topics, word2idx, shap_vals)
    save(topic_vals, f"topic_vals_gpt2_{config['dataset']}_{config['topics']}")
    save(word_vals, f"word_vals_gpt2_{config['dataset']}_{config['topics']}")
    save(shap_vals, f"shap_vals_gpt2_{config['dataset']}")


if __name__ == "__main__":
    main()
