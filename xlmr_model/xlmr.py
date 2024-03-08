#!/usr/bin/env python
# coding: utf-8
import argparse
import json
import os
import random
import timeit
import warnings

import numpy as np
import pandas as pd
import torch
import wandb
from transformers import Trainer, TrainingArguments
from transformers import AutoModelForTokenClassification
from transformers import AutoTokenizer
from transformers.integrations import WandbCallback

import pickle

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


torch.manual_seed(3407)
random.seed(3407)
np.random.seed(3407)

warnings.filterwarnings("ignore", category=FutureWarning)


def convert_type(df):
    for i in range(len(df)):
        df["word"].iloc[i] = eval(df["word"].iloc[i])
        df["labels"].iloc[i] = eval(df["labels"].iloc[i])
    return df

def tokenize_and_align_labels(texts, tags):
    # lowercase
    texts = [[x.lower() for x in l] for l in texts]
    tokenized_inputs = tokenizer(
        texts,
        padding=True,
        truncation=True,
        # We use this argument because the texts in our dataset are lists of words (with a label for each word).
        is_split_into_words=True,
    )
    labels = []
    for i, label in enumerate(tags):
        word_ids = tokenized_inputs.word_ids(batch_index=i)
        previous_word_idx = None
        label_ids = []
        for word_idx in word_ids:
            # Special tokens have a word id that is None. We set the label to -100 so they are automatically
            # ignored in the loss function.
            if word_idx is None:
                label_ids.append(-100)
            # We set the label for the first token of each word.
            elif word_idx != previous_word_idx:
                label_ids.append(label_to_id[label[word_idx]])
            # For the other tokens in a word, we set the label to either the current label or -100, depending on
            # the label_all_tokens flag.
            else:
                label_ids.append(-100)
            previous_word_idx = word_idx

        labels.append(label_ids)
    tokenized_inputs["labels"] = labels
    return tokenized_inputs


# create dataset that can be used for training with the huggingface trainer
class OurDataset(torch.utils.data.Dataset):
    def __init__(self, encodings, labels):
        self.encodings = encodings
        self.labels = labels

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item["labels"] = torch.tensor(self.labels[idx])
        return item

    def __len__(self):
        return len(self.labels)


# return the extracted terms given the token level prediction and the original texts
def extract_terms(token_predictions, val_texts):
    extracted_terms = set()
    # go over all predictions
    for i in range(len(token_predictions)):
        pred = token_predictions[i]
        txt = val_texts[i]
        # print(len(pred), len(txt))
        for j in range(len(pred)):
            # if right tag build term and add it to the set otherwise just continue
            # print(pred[j], txt[j])
            if pred[j] == "B":
                term = txt[j]
                for k in range(j + 1, len(pred)):
                    if pred[k] == "I":
                        term += " " + txt[k]
                    else:
                        break
                extracted_terms.add(term)
    return extracted_terms


# compute the metrics TermEval style for Trainer
def compute_metrics(p):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Remove ignored index (special tokens)
    true_predictions = [
        [label_list[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    extracted_terms = extract_terms(true_predictions, val)  # ??????
    extracted_terms = set([item.lower() for item in extracted_terms])
    gold_set = gold_validation  # ??????

    true_pos = extracted_terms.intersection(gold_set)
    recall = len(true_pos) / len(gold_set)
    precision = len(true_pos) / len(extracted_terms)
    f1 = (
        2 * (precision * recall) / (precision + recall)
        if precision + recall != 0
        else 0
    )

    return {
        "precision": precision,
        "recall": recall,
        "f1": f1,
    }


def computeTermEvalMetrics(extracted_terms, gold_df):
    # make lower case cause gold standard is lower case
    extracted_terms = set([item.lower() for item in extracted_terms])
    gold_set = set(gold_df)
    true_pos = extracted_terms.intersection(gold_set)
    recall = round(len(true_pos) * 100 / len(gold_set), 2)
    precision = round(len(true_pos) * 100 / len(extracted_terms), 2)
    if precision + recall == 0:
        fscore = 0
    else:
        fscore = round(2 * (precision * recall) / (precision + recall), 2)

    print("Extracted", len(extracted_terms))
    print("Gold", len(gold_set))
    print("Intersection", len(true_pos))
    print("Precision:", precision)
    print("Recall:", recall)
    print("F1:", fscore)

    print(
        str(len(extracted_terms))
        + " | "
        + str(len(gold_set))
        + " | "
        + str(len(true_pos))
        + " | "
        + str(precision)
        + " & "
        + str(recall)
        + " & "
        + str(fscore)
    )
    return len(extracted_terms), len(gold_set), len(true_pos), precision, recall, fscore


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


if __name__ == "__main__":
    

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-store", "--store", type=str, default="saves", help="store the model"
    )
    parser.add_argument(
        "-preds",
        "--pred_path",
        type=str,
        default="results/preds.json",
        help="save predicted candidate list",
    )
    parser.add_argument(
        "-log", "--log_path", type=str, default="results/logs.json", help="save logs"
    )
    parser.add_argument(
        "-wandb_api_key",
        "--wandb_api_key",
        type=str,
        default="",
        help="wandb api key",
    )

    parser.add_argument(
        "-model_name",
        "--model_name",
        type=str,
        default="xlm-roberta-base",
        help="model name",
    )
    parser.add_argument(
        "-use_fast_tokenizer",
        "--use_fast_tokenizer",
        type=bool,
        default=True,
        help="use fast tokenizer",
    )

    args = parser.parse_args()
    args.n_gpu = torch.cuda.device_count()
    args.device = device

    start = timeit.default_timer()
    os.makedirs(args.store, exist_ok=True)
    os.makedirs("results", exist_ok=True)

    path = "./data/annotations/sequential_annotations/iob_annotations_sents_wo_empty/"

    train_texts, train_tags = get_dataset(os.path.join(path, "train"))
    val_texts, val_tags = get_dataset(os.path.join(path, "val"))
    model_name = args.model_name  # This should be either "xlm-roberta-base", "xlm-roberta-large", "roberta-base", or "roberta-large"

    test_texts, test_tags = get_dataset(os.path.join(path, "test"))

    gold_set_for_validation = set(
        pd.read_csv(
            "./data/annotations/unique_annotations_lists/val_unique_terms.tsv",
            delimiter="\t",
            quoting=3,
            names=["Term", "Label"],
        )["Term"].tolist()
    )
    gold_set_for_test = set(
        pd.read_csv(
            "./data/annotations/unique_annotations_lists/test_unique_terms.tsv",
            delimiter="\t",
            quoting=3,
            names=["Term", "Label"],
        )["Term"].tolist()
    )

    tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=args.use_fast_tokenizer, add_prefix_space=True)

    label_list = ["O", "B", "I"]
    label_to_id = {l: i for i, l in enumerate(label_list)}
    num_labels = len(label_list)

    train_input_and_labels = tokenize_and_align_labels(train_texts, train_tags)
    val_input_and_labels = tokenize_and_align_labels(val_texts, val_tags)
    test_input_and_labels = tokenize_and_align_labels(test_texts, test_tags)

    train_dataset = OurDataset(train_input_and_labels, train_input_and_labels["labels"])
    val_dataset = OurDataset(val_input_and_labels, val_input_and_labels["labels"])
    test_dataset = OurDataset(test_input_and_labels, test_input_and_labels["labels"])

    training_args = TrainingArguments(
        output_dir=args.store,
        num_train_epochs=50,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        learning_rate=2e-5,
        logging_dir="./logs",
        evaluation_strategy="steps",
        eval_steps=500,
        load_best_model_at_end=True,
        metric_for_best_model="f1",
        greater_is_better=True,
    )

    model = AutoModelForTokenClassification.from_pretrained(model_name, num_labels=num_labels)

    wandb.login(key=args.wandb_api_key)
    wandb.init(project="coastal_term_extraction_xlmr",
               config={
                   "model_name": model_name,
               })

    val = val_texts
    gold_validation = gold_set_for_validation

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        tokenizer=tokenizer,
        compute_metrics=compute_metrics,
        callbacks=[WandbCallback()],
    )

    trainer.train()
    torch.save(trainer.model.state_dict(), f"{args.store}/best_{model_name.split('/')[1]}_weights.pt")

    val = test_texts
    gold_validation = gold_set_for_test
    test_predictions, test_labels, test_metrics = trainer.predict(test_dataset)
    test_predictions = np.argmax(test_predictions, axis=2)

    true_test_predictions = [
        [label_list[p] for (p, l) in zip(test_prediction, test_label) if l != -100]
        for test_prediction, test_label in zip(test_predictions, test_labels)
    ]

    test_extracted_terms = extract_terms(true_test_predictions, test_texts)

    extracted, gold, true_pos, precision, recall, fscore = computeTermEvalMetrics(
        test_extracted_terms, set(gold_set_for_test)
    )

    with open(args.log_path, "w") as f:
        f.write(json.dumps([[extracted, gold, true_pos, precision, recall, fscore]]))

    with open(args.pred_path, "w") as f:
        f.write(json.dumps([list(test_extracted_terms)]))

    stop = timeit.default_timer()
    print("Time: ", stop - start)
