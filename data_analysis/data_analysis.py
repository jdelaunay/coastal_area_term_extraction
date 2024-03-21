import os
import numpy as np
from collections import Counter
from nltk.stem import WordNetLemmatizer
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from wordcloud import WordCloud


lemmatizer = WordNetLemmatizer()


# Function to count the number of files in a directory
def count_files(directory):
    return len(os.listdir(directory))


# Function to load the data and check the vocabulary size and type-to-token ratio
def basic_analysis(directory):
    vocab = Counter()
    total_tokens = 0

    for filename in os.listdir(directory):
        with open(os.path.join(directory, filename), "r") as file:
            for line in file:
                tokens = line.strip().split()
                vocab.update(tokens)
                total_tokens += len(tokens)

    vocab_size = len(vocab)
    type_to_token_ratio = vocab_size / total_tokens

    print(f"Directory: {directory}")
    print(f"Vocabulary size: {vocab_size}")
    print(f"Total tokens: {total_tokens}")
    print(f"Type-to-token ratio: {type_to_token_ratio}")


# Function to count terms in a file
def count_terms(file_path):
    with open(file_path, "r") as file:
        terms = [line.split("\t")[0] for line in file]
    return len(set(terms))


# Function to find common terms in multiple files
def common_terms(train_file, val_file, test_file):
    with open(train_file, "r") as file:
        train_terms = set(line.split("\t")[0] for line in file)
    with open(val_file, "r") as file:
        val_terms = set(line.split("\t")[0] for line in file)
    with open(test_file, "r") as file:
        test_terms = set(line.split("\t")[0] for line in file)

    train_val_common = train_terms.intersection(val_terms)
    val_test_common = val_terms.intersection(test_terms)
    train_test_common = train_terms.intersection(test_terms)
    all_common = train_terms.intersection(val_terms, test_terms)

    return train_val_common, val_test_common, train_test_common, all_common


# Function to count the total number of annotated terms in the iob annotations
def count_total_annotated_terms(directory):
    total_annotated_terms = 0
    for filename in os.listdir(directory):
        if filename.endswith(".tsv"):
            with open(os.path.join(directory, filename), "r") as file:
                for line in file:
                    if line.split("\t")[1].strip() == "B":
                        total_annotated_terms += 1
    return total_annotated_terms


def get_terms_from_iob_file(file_path):
    terms = []
    term = None
    try:
        if file_path.endswith(".tsv"):  # assuming the files are .tsv
            df = pd.read_csv(file_path, sep="\t", header=None, quoting=3)
            for index, row in df.iterrows():
                if row[1] == "O":
                    if term:
                        terms.append(term)
                        term = None
                elif row[1] == "B":
                    if term:
                        terms.append(term)
                    term = row[0]
                elif row[1] == "I":
                    term += " " + row[0]
    except Exception as e:
        print(e)
        print(f"Error processing {file_path}")
    return [lemmatizer.lemmatize(term) for term in terms]


def get_labels_from_iob2_files(file_path):
    labels = []
    label = None
    try:
        if file_path.endswith(".tsv"):  # assuming the files are .tsv
            df = pd.read_csv(file_path, sep="\t", header=None, quoting=3)
            for index, row in df.iterrows():
                if row[1] == "O":
                    if label:
                        labels.append(label)
                        label = None
                elif "B" in row[1]:
                    if label:
                        labels.append(label)
                    label = row[1].split("-")[1]
    except Exception as e:
        print(e)
        print(f"Error processing {file_path}")
    return labels


def get_terms_frequencies(train_iob_path, val_iob_path, test_iob_path, n=100):
    paths = [train_iob_path, val_iob_path, test_iob_path]
    terms = []
    for path in paths:
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r") as _:
                file_terms = get_terms_from_iob_file(os.path.join(path, file))
                new_terms = [term.lower() for term in file_terms]
                new_terms = [
                    " ".join([lemmatizer.lemmatize(word) for word in term.split()])
                    for term in new_terms
                ]
                terms.extend(new_terms)

    return Counter(terms)


def get_labels_frequencies(train_iob_path, val_iob_path, test_iob_path):
    paths = [train_iob_path, val_iob_path, test_iob_path]
    labels = []
    for path in paths:
        for file in os.listdir(path):
            with open(os.path.join(path, file), "r") as _:
                file_labels = get_labels_from_iob2_files(os.path.join(path, file))
                labels.extend(file_labels)
    return Counter(labels)


def get_top_n(terms_counter, n=100):
    return dict(terms_counter.most_common(n))


def plot_wordcloud(tuples_dict, save_path, title="Wordcloud of lemmatized terms"):
    wordcloud = WordCloud(
        width=800, height=400, background_color="white"
    ).generate_from_frequencies(tuples_dict)
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation="bilinear")
    plt.axis("off")
    plt.title(title)
    # Save the figure in img/
    plt.savefig(save_path)
    plt.show()


# Function to count the number of unique terms in total between train, test and val
def count_unique_terms(train_file, val_file, test_file):
    with open(train_file, "r") as file:
        train_terms = set(line.split("\t")[0] for line in file)
    with open(val_file, "r") as file:
        val_terms = set(line.split("\t")[0] for line in file)
    with open(test_file, "r") as file:
        test_terms = set(line.split("\t")[0] for line in file)
    total_unique_terms = train_terms.union(val_terms, test_terms)
    return len(total_unique_terms)


# Function to count unique lemmatized terms
def count_unique_lemmatized_terms(train_file, val_file, test_file):

    with open(train_file, "r") as file:
        train_terms = set(
            [
                " ".join([lemmatizer.lemmatize(word.lower()) for word in term.split()])
                for term in file.read().split("\n")
            ]
        )

    with open(val_file, "r") as file:
        val_terms = set(
            [
                " ".join([lemmatizer.lemmatize(word.lower()) for word in term.split()])
                for term in file.read().split("\n")
            ]
        )

    with open(test_file, "r") as file:
        test_terms = set(
            [
                " ".join([lemmatizer.lemmatize(word.lower()) for word in term.split()])
                for term in file.read().split("\n")
            ]
        )

    total_unique_terms = train_terms.union(val_terms, test_terms)
    return len(total_unique_terms)


def stats_terms_counter(terms_counter):
    # Calculate the median and mean of term occurrences
    values = list(terms_counter.values())
    median = np.median(values)
    mean = np.mean(values)
    first_quartile = np.percentile(values, 25)
    third_quartile = np.percentile(values, 75)

    print(f"First quartile of term occurrences: {first_quartile}")
    print(f"Third quartile of term occurrences: {third_quartile}")
    print(f"Median of term occurrences: {median}")
    print(f"Mean of term occurrences: {mean}")


def print_stats(base_path):
    texts_tokenized_dir = os.path.join(base_path, "texts_tokenized")
    sents_tokenized_wo_empty_dir = os.path.join(base_path, "sents_tokenized_wo_empty")
    files = [
        os.path.join(
            base_path, "annotations/unique_annotations_lists/train_unique_terms.tsv"
        ),
        os.path.join(
            base_path, "annotations/unique_annotations_lists/val_unique_terms.tsv"
        ),
        os.path.join(
            base_path, "annotations/unique_annotations_lists/test_unique_terms.tsv"
        ),
    ]
    # Count the number of files in the directories
    num_files_texts_tokenized = count_files(texts_tokenized_dir)
    num_files_sents_tokenized_wo_empty = count_files(sents_tokenized_wo_empty_dir)

    print(f"Number of files in texts_tokenized: {num_files_texts_tokenized}")
    print(
        f"Number of files in sents_tokenized_wo_empty: {num_files_sents_tokenized_wo_empty}"
    )

    # Load the data and analyze
    basic_analysis(sents_tokenized_wo_empty_dir)

    # Count terms in files
    terms_count = {file: count_terms(file) for file in files}
    print(terms_count)
    unique_annotations_path = os.path.join(
        base_path, "annotations/unique_annotations_lists"
    )
    # Find common terms
    train_val_common, val_test_common, train_test_common, all_common = common_terms(
        os.path.join(unique_annotations_path, "train_unique_terms.tsv"),
        os.path.join(unique_annotations_path, "val_unique_terms.tsv"),
        os.path.join(unique_annotations_path, "test_unique_terms.tsv"),
    )
    print(f"Number of terms common to all splits: {len(all_common)}")
    print(f"Number of common terms between train and val: {len(train_val_common)}")
    print(f"Number of common terms between val and test: {len(val_test_common)}")
    print(f"Number of common terms between train and test: {len(train_test_common)}")
    print(f"Number of unique terms in total: {count_unique_terms(*files)}")
    print(
        f"Number of unique lemmatized terms in total: {count_unique_lemmatized_terms(*files)}"
    )

    iob_path = os.path.join(
        base_path, "annotations/sequential_annotations/iob_annotations_sents_wo_empty"
    )
    iob_dirs = [
        os.path.join(iob_path, "train"),
        os.path.join(iob_path, "val"),
        os.path.join(iob_path, "test"),
    ]
    total_annotated_terms = sum(
        count_total_annotated_terms(directory) for directory in iob_dirs
    )
    print(f"Total number of annotated terms: {total_annotated_terms}")
    terms_counter = get_terms_frequencies(*iob_dirs, n=100)
    stats_terms_counter(terms_counter)
    top_n = get_top_n(terms_counter, n=100)
    # plot_wordcloud(top_n, f"img/{base_path.split('/')[-2]}_wordcloud.png")

    iob2_path = os.path.join(
        base_path, "annotations/sequential_annotations/iob2_annotations_sents_wo_empty"
    )
    iob2_dirs = [
        os.path.join(iob2_path, "train"),
        os.path.join(iob2_path, "val"),
        os.path.join(iob2_path, "test"),
    ]
    labels_counter = get_labels_frequencies(*iob2_dirs)
    print(f"Number of occurence of each label: {labels_counter}")

    return top_n


def terms_in_common(kb_recommended_path, human_recommended_path):
    kb_uniques = os.path.join(
        kb_recommended_path, "annotations/unique_annotations_lists/en_terms.tsv"
    )
    human_uniques = os.path.join(
        human_recommended_path, "annotations/unique_annotations_lists/en_terms.tsv"
    )
    # Load terms from both files
    kb_terms = pd.read_csv(kb_uniques, sep="\t", header=None)[0].tolist()
    human_terms = pd.read_csv(human_uniques, sep="\t", header=None)[0].tolist()

    # Find common terms
    common_terms = set(kb_terms).intersection(human_terms)

    # Print the number of common terms
    print(f"Number of common terms: {len(common_terms)}")

    # Load lemmatized terms from both files
    kb_lemmatized_terms = [lemmatizer.lemmatize(term) for term in kb_terms]
    kb_lemmatized_terms = [
        " ".join([lemmatizer.lemmatize(word) for word in term.split()])
        for term in kb_terms
    ]
    human_lemmatized_terms = [
        " ".join([lemmatizer.lemmatize(word) for word in term.split()])
        for term in human_terms
    ]
    # Find common lemmatized terms
    common_lemmatized_terms = set(kb_lemmatized_terms).intersection(
        human_lemmatized_terms
    )
    # Print the number of common lemmatized terms
    print(f"Number of common lemmatized terms: {len(common_lemmatized_terms)}")


if __name__ == "__main__":
    os.makedirs("img", exist_ok=True)
    # Define directories and files
    print("-------- KB --------")
    print("======================")
    kb_top_n = print_stats("./data/kb/")
    print("-------- HUMAN --------")
    print("======================")
    human_top_n = print_stats("./data/human/")
    print("-------- TERMS IN COMMON --------")
    print("======================")
    terms_in_common("./data/kb/", "./data/human/")
    # Check the number of terms in common between the two top n
    common_top_n_terms = set(kb_top_n.keys()).intersection(set(human_top_n.keys()))
    print(
        f"Number of terms in common in top n {len(kb_top_n)}: {len(common_top_n_terms)}"
    )
