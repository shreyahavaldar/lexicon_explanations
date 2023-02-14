import torch
import os
import torch
import numpy as np
import scipy as sp
import shap
import transformers
from datasets import load_dataset, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.pair_data import LIWCWordData
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import gensim
from gensim.corpora.dictionary import Dictionary
from gensim.models import LdaModel
import nltk
nltk.download('stopwords')
nltk.download('wordnet')
from nltk.corpus import stopwords
from nltk.stem.wordnet import WordNetLemmatizer
from gensim.models import Phrases
import pprint


def save(obj, name):
    base_path = Path(__file__).parent
    torch.save(obj, base_path / "../output" /  (name + ".pt"))

def load(name):
    base_path = Path(__file__).parent
    if os.path.isfile(base_path / "../output" / (name + ".pt")):
        return torch.load(base_path / "../output" / (name + ".pt"))
    else:
        return None

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
    topic_values = np.zeros(topics.shape[0])
    topics_z = np.concatenate([topics, np.zeros((topics.shape[0], 1))], axis=1)
    for tok, val in zip(tokens, shap_values):
        topic_values += np.array([val * topics_z[i, word2idx.get(tok, -1)]
                                 for i in range(len(topics_z))])
    return topic_values


def sort_shap(shap_values, feature_names):
    sort_idx = np.argsort(shap_values)
    return shap_values[sort_idx], feature_names[sort_idx]

def get_topics(config, data):
    if config["topics"] == "liwc":
        return get_liwc_topics()
    elif config["topics"] == "lda":
        stop_words = stopwords.words('english')
        data = [gensim.utils.simple_preprocess(s, deacc=True) for s in data]

        # Lemmatize all words in documents.
        lemmatizer = WordNetLemmatizer()
        data = [[lemmatizer.lemmatize(token) for token in doc] for doc in data]

        # bigram = Phrases(data, min_count=20)
        # for idx in range(len(data)):
        #     for token in bigram[data[idx]]:
        #         if '_' in token:
        #             # Token is a bigram, add to document.
        #             data[idx].append(token)

        # Create a corpus from a list of texts
        common_dictionary = Dictionary(data)
        common_dictionary.filter_n_most_frequent(100)
        common_dictionary.filter_extremes(no_below=20, no_above=0.6)
        print(len(common_dictionary))

        common_corpus = [common_dictionary.doc2bow(text) for text in data]
        common_corpus = [d for d in common_corpus if len(d) > 0]
        cnt = 0
        for doc in common_corpus:
            if len(doc) == 0:
                cnt += 1
        print(len(common_corpus), cnt)

        temp = common_dictionary[0]  # This is only to "load" the dictionary.
        id2word = common_dictionary.id2token


        # Train the model on the corpus.
        lda = LdaModel(corpus=common_corpus, id2word=id2word, num_topics=30, passes=20, iterations=400)
        print(lda.get_topics())
        pp = pprint.PrettyPrinter(indent=4)
        pp.pprint(lda.print_topics(num_words=20))

        raise NotImplementedError
    else:
        raise NotImplementedError

def get_liwc_topics():
    base_path = Path(__file__).parent
    feature_groups = LIWCWordData(
        base_path / "../data/LIWC2015_processed.csv",
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
    explainer = shap.Explainer(model)
    shap_values = explainer(data)

    word_vals = []
    topic_vals = []
    for i in tqdm(range(shap_values.values.shape[0])):
        values, words = word_shap(shap_values.data[i], shap_values.values[i][:, -1])
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
    base_path = Path(__file__).parent
    if dataset_name == "sst2":
        sst2_train = load_dataset("sst2", split="train")
        sst2_val = load_dataset("sst2", split="validation")
        return sst2_train, sst2_val
    elif dataset_name == "cola":
        cola_train = load_dataset("glue", "cola", split="train")
        cola_val = load_dataset("glue", "cola", split="validation")
        return cola_train, cola_val
    elif dataset_name == "tweet":
        cola_train = load_dataset("ought/raft", "tweet_eval_hate", split="train")
        cola_train = cola_train.rename_column("Tweet", "sentence")
        cola_train = cola_train.rename_column("Label", "labels")
        cola_val = load_dataset("ought/raft", "tweet_eval_hate", split="test")
        cola_val = cola_val.rename_column("Tweet", "sentence")
        cola_val = cola_val.rename_column("Label", "labels")
        return cola_train, cola_val
    elif dataset_name == "emobank":
        emobank = pd.read_csv(base_path / '../data/emobank.csv')
        emobank['labels'] = emobank[['V', 'A', 'D']].sum(axis=1) / 3
        emobank = emobank.rename({'text': 'sentence'}, axis=1)
        emobank_train = Dataset.from_pandas(emobank[emobank["split"] == "train"][["sentence", "labels"]])
        emobank_dev = Dataset.from_pandas(emobank[emobank["split"] == "dev"][["sentence", "labels"]])
        return emobank_train, emobank_dev
    elif dataset_name == "polite":
        polite = pd.read_csv(base_path / '../data/wikipedia.annotated.csv')
        raise NotImplementedError
    else:
        raise NotImplementedError