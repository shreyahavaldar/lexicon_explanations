import torch
import numpy as np
import scipy as sp
import shap
from datasets import load_dataset, Dataset
import transformers
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.pair_data import LIWCWordData
from src.utils import save, load
from src.train import train
from tqdm import tqdm
import pandas as pd


def word_shap(token_strs, shap_values):
    vals = [0.0]
    toks = [""]
    for tok, val in zip(token_strs, shap_values):
        if tok.startswith(" "):
            toks[-1] = toks[-1].strip()
            toks.append("")
            vals.append(0.0)

        toks[-1] = toks[-1] + tok
        vals[-1] = vals[-1] + val
        if tok.endswith(" "):
            toks[-1] = toks[-1].strip()
            toks.append("")
            vals.append(0.0)
    return vals, toks


def topic_shap(tokens, word2idx, topics, shap_values):
    print(shap_values[0])
    print(topics.shape)
    topic_values = np.zeros(topics.shape[0])
    for tok, val in zip(tokens, shap_values):
        if topics[56, word2idx.get(tok, 0)] > 0:
            print("RELIG word:", tok)
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
    topics = topics / np.sum(topics, axis=0, keepdims=True)
    return topics, topic_names, word2idx


def get_topic_shap(model, data, topics, word2idx):
    # def f(x):
    #     tv = torch.tensor([tokenizer.encode(v,
    #                                         padding='max_length',
    #                                         max_length=500,
    #                                         truncation=True) for v in x]).cuda()
    #     outputs = model(tv)[0].detach().cpu().numpy()
    #     scores = (np.exp(outputs).T / np.exp(outputs).sum(-1)).T
    #     val = sp.special.logit(scores[:, 1])  # use one vs rest logit units
    #     return val

    # tokens = [
    #     tokenizer.encode(
    #         v,
    #         truncation=True) for v in data]
    explainer = shap.Explainer(model)
    shap_values = explainer(data, fixed_context=1)

    word_vals = []
    topic_vals = []
    for i in tqdm(range(shap_values.values.shape[0])):
        print(shap_values.data[i])
        values, words = word_shap(shap_values.data[i], shap_values.values[i][:, 1])
        topic_values = topic_shap(words, word2idx, topics, values)
        topic_vals.append(topic_values)
        word_vals.append(values)
    topic_vals = np.stack(topic_vals, axis=0)
    return topic_vals, word_vals


def load_models(config):
    dataset_name = config["dataset"]
    if dataset_name == "sst2" or dataset_name == "cola":
        tokenizer1 = AutoTokenizer.from_pretrained(
            "roberta-base")
        model1 = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base").cuda()
        pred1 = transformers.TextClassificationPipeline(
            model=model1, tokenizer=tokenizer1, device=0, top_k=None)

        tokenizer2 = AutoTokenizer.from_pretrained(
            "gpt2")
        model2 = AutoModelForSequenceClassification.from_pretrained(
            "gpt2").cuda()
        tokenizer2.pad_token = tokenizer2.eos_token
        model2.config.pad_token_id = tokenizer2.pad_token_id
        pred2 = transformers.TextClassificationPipeline(
            model=model2, tokenizer=tokenizer2, device=0, top_k=None)

        return pred1, pred2
    elif dataset_name == "emobank":
        tokenizer1 = AutoTokenizer.from_pretrained(
            "roberta-base")
        model1 = AutoModelForSequenceClassification.from_pretrained(
            "roberta-base", num_labels=1).cuda()
        pred1 = transformers.TextClassificationPipeline(
            model=model1, tokenizer=tokenizer1, device=0, top_k=None)

        tokenizer2 = AutoTokenizer.from_pretrained(
            "gpt2")
        model2 = AutoModelForSequenceClassification.from_pretrained(
            "gpt2", num_labels=1).cuda()
        tokenizer2.pad_token = tokenizer2.eos_token
        model2.config.pad_token_id = tokenizer2.pad_token_id
        pred2 = transformers.TextClassificationPipeline(
            model=model2, tokenizer=tokenizer2, device=0, top_k=None)

        return pred1, pred2
    else:
        raise NotImplementedError

def load_data(config):
    dataset_name = config["dataset"]
    if dataset_name == "sst2":
        sst2_train = load_dataset("sst2", split="train")
        sst2_val = load_dataset("sst2", split="validation")
        return sst2_train, sst2_val
    elif dataset_name == "cola":
        cola_train = load_dataset("glue", "cola", split="train")
        cola_val = load_dataset("glue", "cola", split="validation")
        return cola_train, cola_val
    elif dataset_name == "emobank":
        emobank = pd.read_csv('data/emobank.csv')
        emobank['label'] = emobank[['V', 'A', 'D']].sum(axis=1)
        emobank_train = Dataset.from_pandas(emobank[emobank["split"] == "train"][["text", "label"]])
        emobank_dev = Dataset.from_pandas(emobank[emobank["split"] == "dev"][["text", "label"]])
        return emobank_train, emobank_dev
    elif dataset_name == "polite":
        polite = pd.read_csv('data/wikipedia.annotated.csv')
    else:
        raise NotImplementedError


def main():
    config = {"dataset": "cola"}
    model1, model2 = load_models(config)

    sst2_train = load_dataset("sst2", split="train")
    sst2_val = load_dataset("sst2", split="validation")
    x = [sst2_val[i]['sentence'] for i in range(len(sst2_val))]
    topics, topic_names, word2idx = get_liwc_topics()

    train(config, model1, sst2_train, sst2_val)
    topic_vals, word_vals = get_topic_shap(model1, x, topics, word2idx)
    save(topic_vals, f"topic_vals_roberta_{config['dataset']}")
    save(word_vals, f"word_vals_roberta_{config['dataset']}")
    del model1.model
    del model1
    torch.cuda.empty_cache()

    train(config, model2, sst2_train, sst2_val)
    topic_vals, word_vals = get_topic_shap(model2, x, topics, word2idx)
    save(topic_vals, f"topic_vals_gpt2_{config['dataset']}")
    save(word_vals, f"word_vals_gpt2_{config['dataset']}")


if __name__ == "__main__":
    main()
