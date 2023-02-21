import torch
import os
import torch
import numpy as np
import shap
import transformers
from datasets import load_dataset, Dataset, Sequence, Value, Features
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from src.pair_data import LIWCWordData
from tqdm import tqdm
import pandas as pd
from pathlib import Path
import nltk
from sklearn.model_selection import train_test_split
import string
from sklearn.preprocessing import MultiLabelBinarizer

from octis.preprocessing.preprocessing import Preprocessing
from octis.dataset.dataset import Dataset as octDataset
from octis.models.LDA import LDA
from octis.models.NeuralLDA import NeuralLDA
from octis.models.ETM import ETM
from octis.models.CTM import CTM
from octis.optimization.optimizer import Optimizer
from skopt.space.space import Real, Categorical, Integer
from octis.evaluation_metrics.coherence_metrics import Coherence
import csv


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
    # Add an extra topic for words not in the topic model
    topic_values = np.zeros(topics.shape[0])
    topics_z = np.concatenate([topics, np.zeros((topics.shape[0], 1))], axis=1)
    for tok, val in zip(tokens, shap_values):
        topic_values += np.array([val * topics_z[i, word2idx.get(tok, -1)]
                                 for i in range(len(topics_z))])
    no_topic = 0.0
    for tok, val in zip(tokens, shap_values):
        no_topic += val * (word2idx.get(tok, -1) == -1)
    return np.concatenate([topic_values, np.array([no_topic])])


def sort_shap(shap_values, feature_names):
    sort_idx = np.argsort(shap_values)
    return shap_values[sort_idx], feature_names[sort_idx]

def get_topics(config, data):
    if config["topics"] == "liwc":
        return get_liwc_topics()
    elif config["topics"] == "ctm" or config["topics"] == "neurallda":
        base_path = Path(__file__).parent
        data_name = f"{config['dataset']}_raw.txt"
        
        if not os.path.isfile(base_path / "../data" / data_name):
            with open(base_path / ("../data/" + data_name), "w", newline="") as csvfile:
                writer = csv.writer(csvfile, delimiter="\t", escapechar="\\")
                for line in data:
                    writer.writerow([line])

        if not os.path.exists(base_path / "../data" / (config["dataset"] + "_preprocessed")):
            preprocessor = Preprocessing(
                num_processes=None,
                vocabulary=None,
                max_features=None,
                remove_punctuation=True,
                punctuation=string.punctuation,
                lemmatize=True,
                stopword_list="english",
                min_chars=1,
                min_words_docs=0,
                max_df=0.95)
            dataset = preprocessor.preprocess_dataset(documents_path=(base_path / ("../data/" + data_name)))
            dataset.save("data/" + config["dataset"] + "_preprocessed")

        dataset = octDataset()
        dataset.load_custom_dataset_from_folder("data/" + config["dataset"] + "_preprocessed")
        # npmi = Coherence(texts=dataset.get_corpus())

        if config["topics"] == "ctm":
            model = CTM(num_topics=10, num_epochs=35, inference_type='zeroshot', bert_model="bert-base-nli-mean-tokens")
        elif config["topics"] == "neurallda":
            model = NeuralLDA(num_topics=35, lr=0.001)

        # search_space = {"num_layers": Categorical({1, 2, 3}), 
        #         "num_neurons": Categorical({100, 200, 300}),
        #         "activation": Categorical({'sigmoid', 'relu', 'softplus'}), 
        #         "dropout": Real(0.0, 0.95)
        # }
        model_output = model.train_model(dataset)
        print(model_output)
        word2idx = {word: i for i, word in enumerate(dataset.get_vocabulary())}
        topics = model_output["topic-word-matrix"]
        print(model_output["topics"][0][:5])
        print([dataset.get_vocabulary()[i] for i in np.argsort(-topics[0, :])[:5]])
        print([model.model.train_data.idx2token[i] for i in np.argsort(-topics[0, :])[:5]])
        topics = topics / np.sum(topics, axis=0, keepdims=True)
        return topics, word2idx
    
    elif config["topics"] == "lda":
        base_path = Path(__file__).parent
        topics_matrix_df = pd.read_csv(base_path / ("../data/processed_LDA_files/" + config["dataset"] + ".csv"))
        word2idx = dict(zip(topics_matrix_df["words"], range(len(topics_matrix_df["words"]))))
        topics_matrix_df.drop(columns=["words"], inplace=True)
        topics_matrix_df = topics_matrix_df.T
        topics = topics_matrix_df.to_numpy()
        topics = topics / np.sum(topics, axis=0, keepdims=True)
        return topics, word2idx

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


def get_topic_shap(model, data, topics, word2idx, shap_values=None):
    if shap_values == None:
        explainer = shap.Explainer(model)
        shap_values = explainer(data, fixed_context=1).values

    word_vals = []
    topic_vals = []
    for i in tqdm(range(shap_values.shape[0])):
        values, words = word_shap(data[i], shap_values[i][:, -1])
        topic_values = topic_shap(words, word2idx, topics, values)
        topic_vals.append(topic_values)
        word_vals.append(values)
    topic_vals = np.stack(topic_vals, axis=0)
    return shap_values, topic_vals, word_vals


def load_models(config):
    dataset_name = config["dataset"]
    if dataset_name == "goemotions":
        num_labels = 28
        problem_type = "multi_label_classification"
    elif dataset_name == "blog":
        num_labels = 45
        problem_type = "multi_label_classification"
    elif dataset_name == "polite":
        num_labels = 1
        problem_type = "regression"
    else:
        num_labels = 2
        problem_type = "single_label_classification"


    tokenizer1 = AutoTokenizer.from_pretrained(
        "distilroberta-base")
    model1 = AutoModelForSequenceClassification.from_pretrained(
        "distilroberta-base", num_labels=num_labels, problem_type=problem_type).cuda()
    pred1 = transformers.TextClassificationPipeline(
        model=model1, tokenizer=tokenizer1, device=0, top_k=None)

    tokenizer2 = AutoTokenizer.from_pretrained(
        "gpt2")
    model2 = AutoModelForSequenceClassification.from_pretrained(
        "gpt2", num_labels=num_labels, problem_type=problem_type).cuda()
    tokenizer2.pad_token = tokenizer2.eos_token
    model2.config.pad_token_id = tokenizer2.pad_token_id
    pred2 = transformers.TextClassificationPipeline(
        model=model2, tokenizer=tokenizer2, device=0, top_k=None)

    return pred1, pred2

def load_data(config):
    dataset_name = config["dataset"]
    base_path = Path(__file__).parent
    if dataset_name == "sst2":
        sst2_train = load_dataset("glue", "sst2", split="train")
        sst2_val = load_dataset("glue", "sst2", split="validation")
        sst2_test = load_dataset("glue", "sst2", split="test")
        return sst2_train, sst2_val, sst2_test
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
    elif dataset_name == "goemotions":
        goemotions_train = load_dataset("go_emotions", "simplified", split="train")
        goemotions_train = goemotions_train.rename_column("text", "sentence")
        goemotions_val = load_dataset("go_emotions", "simplified", split="validation")
        goemotions_val = goemotions_val.rename_column("text", "sentence")
        goemotions_test = load_dataset("go_emotions", "simplified", split="test")
        goemotions_test = goemotions_test.rename_column("text", "sentence")
        def to_tensor(x):
            y = torch.zeros((28)).float()
            y[x["labels"]] = 1.0
            x["labels"] = y
            return x
        goemotions_train = goemotions_train.map(to_tensor)
        goemotions_val = goemotions_val.map(to_tensor)
        goemotions_test = goemotions_test.map(to_tensor)
        new_features = goemotions_train.features.copy()
        new_features["labels"] = Sequence(Value("float32"))
        goemotions_train = goemotions_train.cast(new_features)
        goemotions_val = goemotions_val.cast(new_features)
        goemotions_test = goemotions_test.cast(new_features)
        
        return goemotions_train, goemotions_val, goemotions_test
    elif dataset_name == "blog":
        blog_train = load_dataset("blog_authorship_corpus", split="train")
        blog_train = blog_train.rename_column("text", "sentence")
        blog_val = load_dataset("blog_authorship_corpus", split="validation")
        blog_val = blog_val.rename_column("text", "sentence")
        def get_raw_labels(x):
            age_group = "10s"
            if x["age"] >= 33:
                age_group = "30s"
            elif x["age"] >= 23:
                age_group = "20s"
            x["raw_labels"] = [age_group, x["gender"], x["job"]]
            return x

        blog_train = blog_train.map(get_raw_labels)
        blog_val = blog_val.map(get_raw_labels)
        mlb = MultiLabelBinarizer()
        train_labels = list(mlb.fit_transform([blog["raw_labels"] for blog in blog_train]).astype(float))
        val_labels = list(mlb.transform([blog["raw_labels"] for blog in blog_val]).astype(float))

        blog_train = blog_train.add_column("labels", train_labels)
        blog_val = blog_val.add_column("labels", val_labels)

        print(blog_train[0])
        return blog_train, Dataset.from_list([]), blog_val
    elif dataset_name == "emobank":
        emobank = pd.read_csv(base_path / '../data/emobank.csv')
        emobank['labels'] = emobank[['V', 'A', 'D']].sum(axis=1) / 3
        emobank = emobank.rename({'text': 'sentence'}, axis=1)
        emobank_train = Dataset.from_pandas(emobank[emobank["split"] == "train"][["sentence", "labels"]])
        emobank_dev = Dataset.from_pandas(emobank[emobank["split"] == "dev"][["sentence", "labels"]])
        return emobank_train, emobank_dev
    elif dataset_name == "polite":
        polite = pd.read_csv(base_path / '../data/wikipedia.annotated.csv')
        polite = polite.rename({"Request": "sentence", "Normalized Score": "labels"}, axis=1)
        polite_train, polite_test = train_test_split(polite, test_size=0.2)
        polite_val, polite_test = train_test_split(polite_test, test_size=0.5)
        polite_train = Dataset.from_pandas(polite_train[["sentence", "labels"]])
        polite_val = Dataset.from_pandas(polite_val[["sentence", "labels"]])
        polite_test = Dataset.from_pandas(polite_test[["sentence", "labels"]])
        return polite_train, polite_val, polite_test
    elif dataset_name == "yelp":
        yelp_train = load_dataset("yelp_review_full", split="train")
        yelp_train = yelp_train.rename_column("text", "sentence").rename_column("label", "labels")
        yelp_test = load_dataset("yelp_review_full", split="test")
        yelp_test = yelp_test.rename_column("text", "sentence").rename_column("label", "labels")
        return yelp_train, Dataset.from_list([]), yelp_test
    else:
        raise NotImplementedError

def write_to_db(dataset_name, table_name, splits):
    train= []
    val = []
    test = []
    if("train" in splits):
        data_train = load_dataset(dataset_name, split="train")  
        train = [data_train[i]['text'] for i in tqdm(range(len(data_train)))]
    if("val" in splits):
        data_val = load_dataset(dataset_name, split="validation")
        val = [data_val[i]['text'] for i in tqdm(range(len(data_val)))]
    if("test" in splits):
        data_test = load_dataset(dataset_name, split="test")
        test = [data_test[i]['text'] for i in tqdm(range(len(data_test)))]

    msgs = np.concatenate([train, test, val])

    df = pd.DataFrame(columns=["message_id", "message"])
    message_ids = range(len(msgs))
    df["message"] = msgs
    df["message_id"] = message_ids
    
    db = sqlalchemy.engine.url.URL(drivername='mysql', host='127.0.0.1', database='shreyah', query={'read_default_file': '~/.my.cnf', 'charset':'utf8mb4'})
    engine = sqlalchemy.create_engine(db)
    df.to_sql(table_name, con=engine, index=False, if_exists='replace', chunksize=50000)

def process_mallet_topics(filepath, numtopics, dataset):
    topics = {}
    words = set(())
    for i in range(numtopics):
        topics[i] = {}

    with open(filepath) as f:
        lines = csv.reader(f, delimiter=',', quotechar='\"')
        for parts in lines:
            if(len(parts) == 6):
                continue
            topic_id = int(parts[0])
            parts = parts[1:]
            for i in range(len(parts)):      
                if(i%2 == 0): #word
                    try:
                        word = parts[i]
                        score = parts[i+1]
                        words.add(word)
                    except:
                        print("error")
                (topics[topic_id])[word] = score
    
    words = list(words)
    word_scores = []        
    for w in words:
        scores = []
        for t in topics:
            try:
                score = topics[t][w]
            except(KeyError):
                score = 0
            scores.append(score)
        word_scores.append(scores)
    
    lda_scores = pd.DataFrame(data=word_scores, columns=range(numtopics))
    lda_scores["words"] = words
    outfile = "data/processed_LDA_files/" + dataset + ".csv"
    lda_scores.to_csv(outfile, index=False)

    #ensure all columns add to 1
    for i in range(numtopics):
        print(np.sum(np.array(lda_scores[i].astype(float))))

def process_lda():
    process_mallet_topics("LDA/sst_lda_100_30/lda.wordGivenTopic.csv", 30, "sst")
    process_mallet_topics("LDA/blog_lda_100_30/lda.wordGivenTopic.csv", 30, "blog")
    process_mallet_topics("LDA/emotions_lda_100_30/lda.wordGivenTopic.csv", 30, "goemotions")
    process_mallet_topics("LDA/polite_lda_100_30/lda.wordGivenTopic.csv", 30, "polite")
    process_mallet_topics("LDA/yelp_lda_100_30/lda.wordGivenTopic.csv", 30, "yelp")

