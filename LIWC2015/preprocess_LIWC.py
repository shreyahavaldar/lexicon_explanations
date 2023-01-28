import pandas as pd
import numpy as np
import re
from tqdm import tqdm, trange
import json

#return true if base_word and word are a match; ex: (accur*, accurate)
def is_match(base_word, word):
    prefix = base_word.split("*")[0]
    return word.startswith(prefix)

#make a dictionary of base_words --> possible words
def make_dictionary():
    #read in LIWC and english list of words
    LIWC2015 = pd.read_csv("LIWC2015.csv")
    liwc_words = np.unique(LIWC2015["term"])
    all_words = []
    with open("english_words.txt") as f:
        all_words = f.read().splitlines()
    
    #iterate through LIWC and make dictionary
    word_expansion = {}
    for base_word in tqdm(liwc_words):
        if("*" in base_word):
            #preprocess
            if("(" in base_word and ")" in base_word):
                print(base_word)
                base_word=base_word.replace("(", "")
                base_word=base_word.replace(")", "")
                print(base_word)
            word_expansion[base_word] = []
            for word in all_words:
                if(is_match(base_word, word)):
                    word_expansion[base_word].append(word)
    with open('dictionary.txt', 'w') as file:
        file.write(json.dumps(word_expansion))

def preprocess_LIWC():
    with open('dictionary.txt') as f:
        data = f.read()
    dictionary = json.loads(data)  
    LIWC2015 = pd.read_csv("LIWC2015.csv") 

    processed_LIWC = pd.DataFrame(columns = ["term", "category"])
    terms = LIWC2015["term"]
    categories = LIWC2015["category"]
    processed_terms = []
    processed_categories = []
    for i in range(len(terms)):
        term = terms[i]
        category = categories[i]
        if("(" in term and ")" in term):
                term=term.replace("(", "")
                term=term.replace(")", "")
        if("*" in term):
            term_expansion = dictionary[term]
            term_expansion.sort(key=lambda x: len(x))
            # EVENTUALLY: sort by frequency
            for t in term_expansion[:2]:
                print(t)
                processed_terms.append(t)
                processed_categories.append(category)
        else:
            processed_terms.append(term)
            processed_categories.append(category)
    
    print(len(np.unique(LIWC2015["term"])))
    print(len(np.unique(processed_terms)))

    processed_LIWC["term"] = processed_terms
    processed_LIWC["category"] = processed_categories
    processed_LIWC.to_csv("LIWC2015_processed.csv")


preprocess_LIWC()
