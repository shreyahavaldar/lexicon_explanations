import torch
import numpy as np
import scipy as sp
import shap
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.pair_data import LIWCWordData
from src.utils import save, load
from tqdm import tqdm


def word_shap(token_strs, shap_values):
    vals = [0.0]
    toks = [""]
    for tok, val in zip(token_strs, shap_values):
        toks[-1] = toks[-1] + tok
        vals[-1] = vals[-1] + val
        if tok.endswith(" "):
            toks[-1] = toks[-1].strip()
            toks.append("")
            vals.append(0.0)
    return vals, toks


def topic_shap(tokens, word2idx, topics, shap_values):
    topic_values = np.zeros(topics.shape[0])
    for tok, val in zip(tokens, shap_values):
        topic_values += np.array([val * topics[i, word2idx.get(tok, 0)]
                                 for i in range(len(topics))])
    return topic_values


def sort_shap(shap_values, feature_names):
    sort_idx = np.argsort(shap_values)
    return shap_values[sort_idx], feature_names[sort_idx]


def get_liwc_topics():
    feature_groups = LIWCWordData(
        "data/LIWC2015_processed.csv",
        split="train")

    print(feature_groups.groups)
    topic_names = feature_groups.groups
    topics = {name: set() for name in topic_names}
    print(feature_groups[0])
    for word, groups in feature_groups:
        for group in groups:
            topics[group].add(word)
    all_tokens = set().union(*topics.values())
    word2idx = {word: i for i, word in enumerate(all_tokens)}
    topics = np.array([[1.0 if tok in topics[topic_names[i]]
                        else 0.0 for tok in all_tokens] for i in range(len(topic_names))])
    topics = topics / np.sum(topics, axis=1, keepdims=True)
    return topics, topic_names, word2idx


def get_topic_shap(model, tokenizer, data, topics, word2idx):
    def f(x):
        tv = torch.tensor([tokenizer.encode(v,
                                            padding='max_length',
                                            max_length=500,
                                            truncation=True) for v in x]).cuda()
        outputs = model(tv)[0].detach().cpu().numpy()
        scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
        val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
        return val
    tokens = [
        tokenizer.encode(
            v,
            truncation=True) for v in data]
    explainer = shap.Explainer(f, tokenizer)
    shap_values = explainer(data, fixed_context=1)

    topic_vals = []
    for i in tqdm(range(shap_values.values.shape[0])):
        values, words = word_shap(shap_values.data[i], shap_values.values[i])
        # print(words, values)
        topic_values = topic_shap(words, word2idx, topics, values)
        # topic_values, sorted_names = sort_shap(topic_values, topic_names)
        # for i in range(5):
        #     print(f"{sorted_names[i]}: {topic_values[i]}")
        # for i in range(5, 0, -1):
        #     print(f"{sorted_names[-i]}: {topic_values[-i]}")
        topic_vals.append(topic_values)
    topic_vals = np.concatenate(topic_vals, axis=0)
    return topic_vals


def main():
    # Computing on distilbert base
    tokenizer1 = AutoTokenizer.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english")
    model1 = AutoModelForSequenceClassification.from_pretrained(
        "distilbert-base-uncased-finetuned-sst-2-english").cuda()

    sst2 = load_dataset("sst2", split="validation")
    topics, topic_names, word2idx = get_liwc_topics()
    x = [sst2[i]['sentence'] for i in range(len(sst2))]

    topic_vals = get_topic_shap(model1, tokenizer1, x, topics, word2idx)
    save(topic_vals, "topic_vals_distilbertbase_sst2")

    # Computing on roberta large
    tokenizer2 = AutoTokenizer.from_pretrained(
        "philschmid/roberta-large-sst2")
    model2 = AutoModelForSequenceClassification.from_pretrained(
        "philschmid/roberta-large-sst2").cuda()

    topic_vals = get_topic_shap(model2, tokenizer2, x, topics, word2idx)
    save(topic_vals, "topic_vals_robertalarge_sst2")


if __name__ == "__main__":
    main()
