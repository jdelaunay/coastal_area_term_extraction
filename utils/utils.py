import os
import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer

def get_dataset(path):
    files = [
        os.path.join(path, file)
        for file in os.listdir(path)
        if os.path.isfile(os.path.join(path, file))
    ]
    texts, tags = [], []
    for file in files:
        df = pd.read_csv(file, delimiter="\t", quoting=3, names=["word", "labels"])
        texts.append(df["word"].tolist())
        tags.append(df["labels"].tolist())
    return texts, tags

def get_gold_set(path, lemmatized=False):
    gold_set =  set(
        pd.read_csv(
            path, delimiter="\t", quoting=3, names=["Term", "Label"]
        )["Term"].tolist()
    )
    gold_set = set([item.lower() for item in gold_set])
    if lemmatized:
        lemmatizer = WordNetLemmatizer()
        gold_set = set([lemmatizer.lemmatize(term) for term in gold_set])
    return gold_set