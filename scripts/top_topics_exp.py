from src.utils import *
import numpy as np
from sklearn.cluster import KMeans
import argparse


def get_shap_examples(dataset_name):
    config = {"dataset": dataset_name, "topics": "lda"}
    _, model2 = load_models(config)
    explainer = shap.Explainer(model2, padding="max_length", truncation=True, max_length=512)

    data_train, data_val, data_test = load_data(config)

    def tokenize_function(examples):
        output = len(model2.tokenizer.tokenize(examples["sentence"], padding=False, truncation=True, max_length=512)) > 1
        return output

    #  Filter then select only for blog
    if config["dataset"] == "blog":
        indices = list(range(len(data_test)))
        random.seed(316)
        random.shuffle(indices)
        data_test_shap = data_test.filter(tokenize_function)
        data_test_shap = data_test_shap.select(indices[:1000])
    else:
        indices = list(range(len(data_test)))
        random.seed(316)
        random.shuffle(indices)
        data_test_shap = data_test.select(indices[:1000])
        data_test_shap = data_test_shap.filter(tokenize_function)

    x = [data_test_shap[i]['sentence'] for i in range(len(data_test_shap))]
    return x

def load_shap_vals(dataset_name: str, topic_type: str, feature: int):
    config = {"dataset": dataset_name, "topics": topic_type}
    # topics, word2idx = get_topics(config, [])
    shap_vals = load(f"shap_vals_distilroberta_{dataset_name}")
    roberta_topic_vals = load(f"topic_vals_distilroberta_{dataset_name}_{topic_type}")
    roberta_word_vals = load(f"word_vals_distilroberta_{dataset_name}_{topic_type}")
    gpt2_topic_vals = load(f"topic_vals_gpt2_{dataset_name}_{topic_type}")
    gpt2_word_vals = load(f"word_vals_gpt2_{dataset_name}_{topic_type}")

    assert roberta_topic_vals is not None
    assert gpt2_topic_vals is not None
    print(roberta_topic_vals.shape)
    print(gpt2_topic_vals.shape)
    # roberta_topic_vals = roberta_topic_vals[:, :, feature]
    # gpt2_topic_vals = gpt2_topic_vals[:, :, feature]
    roberta_topic_vals = roberta_topic_vals[:, feature]
    gpt2_topic_vals = gpt2_topic_vals[:, feature]

    return shap_vals, roberta_topic_vals, gpt2_topic_vals

def print_global_exp(roberta_topic_vals, gpt2_topic_vals, topic_names):
    # roberta_feat_imp = np.sum(np.abs(roberta_topic_vals), axis=0)
    # gpt2_feat_imp = np.sum(np.abs(gpt2_topic_vals), axis=0)
    roberta_feat_imp = roberta_topic_vals / np.linalg.norm(roberta_topic_vals)
    gpt2_feat_imp = gpt2_topic_vals / np.linalg.norm(gpt2_topic_vals)

    # diff = roberta_feat_imp - (gpt2_feat_imp * np.dot(roberta_feat_imp, gpt2_feat_imp))
    diff = roberta_feat_imp - gpt2_feat_imp
    diff_abs = np.abs(diff)
    # diff = np.abs(roberta_feat_imp - gpt2_feat_imp)
    sort_idx = np.argsort(diff_abs)
    # topic_values, sorted_names = sort_shap(diff_abs, topic_names)
    print("Most diff explanation:")
    for i in range(-1, -6, -1):
        print(f"{topic_names[sort_idx[i]]}: {diff[sort_idx[i]]}")

    print()
    topic_values, sorted_names = sort_shap(roberta_feat_imp, topic_names)
    print("Roberta explanation:")
    for i in range(-1, -6, -1):
        print(f"{sorted_names[i]}: {topic_values[i]}")
    
    print()
    topic_values, sorted_names = sort_shap(gpt2_feat_imp, topic_names)
    print("GPT2 explanation:")
    for i in range(-1, -6, -1):
        print(f"{sorted_names[i]}: {topic_values[i]}")

def get_topic_names(dataset, topic_type):
    if topic_type == "liwc":
        liwc_topics = ['ACHIEVE', 'ADJ', 'ADVERB', 'AFFECT', 'AFFILIATION', 'ANGER', 'ANX', 'ARTICLE',
                'ASSENT', 'AUXVERB', 'BIO', 'BODY', 'CAUSE', 'CERTAIN', 'COGPROC', 'COMPARE',
                'CONJ', 'DEATH', 'DIFFER', 'DISCREP', 'DRIVES', 'FAMILY', 'FEEL', 'FEMALE',
                'FILLER', 'FOCUSFUTURE', 'FOCUSPAST', 'FOCUSPRESENT', 'FRIEND', 'FUNCTION',
                'HEALTH', 'HEAR', 'HOME', 'I', 'INFORMAL', 'INGEST', 'INSIGHT', 'INTERROG',
                'IPRON', 'LEISURE', 'MALE', 'MONEY', 'MOTION', 'NEGATE', 'NEGEMO', 'NETSPEAK',
                'NONFLU', 'NUMBER', 'PERCEPT', 'POSEMO', 'POWER', 'PPRON', 'PREP', 'PRONOUN',
                'QUANT', 'RELATIV', 'RELIG', 'REWARD', 'RISK', 'SAD', 'SEE', 'SEXUAL', 'SHEHE',
                'SOCIAL', 'SPACE', 'SWEAR', 'TENTAT', 'THEY', 'TIME', 'VERB', 'WE', 'WORK', 'YOU']

        if dataset == "goemotions":
            stopword_topics = load("stopword_topic_names_goemotions_liwc")
        elif dataset == "yelp":
            stopword_topics = load("stopword_topic_names_yelp_liwc")
        elif dataset == "blog":
            stopword_topics = load("stopword_topic_names_blog_liwc")
        
        n_liwc = len(liwc_topics)
        for i in range(len(stopword_topics)):
            liwc_topics.append(stopword_topics[n_liwc + i])
        return np.array(liwc_topics + ["punctuation"])
    else:
        if dataset == "goemotions":
            names = []
            stopword_topics = load("stopword_topic_names_goemotions_lda")
            for i in range(531):
                if i in stopword_topics:
                    names.append(stopword_topics[i])
                elif i == 530:
                    names.append("punctuation")
                else:
                    names.append(f"topic_{i}")
            return np.array(names)
        elif dataset == "yelp":
            names = []
            stopword_topics = load("stopword_topic_names_yelp_lda")
            for i in range(175):
                if i in stopword_topics:
                    names.append(stopword_topics[i])
                elif i == 174:
                    names.append("punctuation")
                else:
                    names.append(f"topic_{i}")
            return np.array(names)
        elif dataset == "blog":
            names = []
            stopword_topics = load("stopword_topic_names_blog_lda")
            for i in range(1726):
                if i in stopword_topics:
                    names.append(stopword_topics[i])
                elif i == 1725:
                    names.append("punctuation")
                else:
                    names.append(f"topic_{i}")
            return np.array(names)
    
        
def main(dataset_name):
    topic_types = ["liwc"]
    # x = get_shap_examples(dataset_name)

    for topic_type in topic_types:
        features = []
        feature_names = []
        if dataset_name == "yelp":
            features = list(range(5))
            feature_names = [f"{i+1} star" for i in range(5)]
        elif dataset_name == "blog":
            features = list(range(5))
            feature_names = ["male", "female", "10s", "20s", "30s"]
        elif dataset_name == "goemotions":
            features = list(range(6))
            feature_names = ["anger", "surprise", "disgust", "joy", "fear", "sadness"]

        for feature in features:
            shap_vals, roberta_topic_vals, gpt2_topic_vals = load_shap_vals(dataset_name, topic_type, feature)
            topic_names = get_topic_names(dataset_name, topic_type)
            print(f"Explanations for {dataset_name} with {topic_type} and feature [{feature}: {feature_names[feature]}]")
            print_global_exp(roberta_topic_vals, gpt2_topic_vals, topic_names)
            print()


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument("-d", "--dataset", type=str, help="dataset name", choices=["yelp", "blog", "goemotions"])
    args = argparser.parse_args()
    main(args.dataset)