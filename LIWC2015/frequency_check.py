import pandas as pd
import numpy as np
import re
from tqdm import tqdm, trange
import json

def compare_frequencies():
    
    LIWC_words = np.array(pd.read_csv("LIWC2015.csv")["term"])

    data = []
    with open('frequencies.txt') as f:
        data = f.read().splitlines()
    
    liwc_count = 0
    count = 0
    for d in data:
        word = d.split(" ")[0]
        if(word in LIWC_words):
            liwc_count+=1
        count+=1
    print(liwc_count/count)

compare_frequencies()