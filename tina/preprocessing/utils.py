import csv
import json
from random import sample

import pandas as pd
from datasets import load_dataset
from nltk.tokenize import word_tokenize
from tqdm import tqdm

from tina.utils import ID_TO_LABEL_RTE, ID_TO_LABEL_SNLI_MNLI


def read_jsonl(dataset_path):
    """
    It reads a jsonl file and returns a list of dictionaries

    :param dataset_path: the path to the dataset file
    :return: A list of dictionaries.
    """
    with open(dataset_path, "r", encoding="utf-8") as out:
        jsonl = list(out)
    return [json.loads(i) for i in jsonl]


def extract_sentences(
    json_instance, pos_statmt_key, neg_statmt_key, object_key, repl_keyword
):
    object_word = json_instance[object_key]
    pos_statmt = json_instance[pos_statmt_key][0].replace(repl_keyword, object_word)
    neg_statmt = json_instance[neg_statmt_key][0].replace(repl_keyword, object_word)
    return pos_statmt, neg_statmt


def read_dataset_grammar(dataset_path):
    data = pd.read_csv(dataset_path)
    p = data.values[:, 0].tolist()
    h = data.values[:, 1].tolist()
    n_p = data.values[:, 3].tolist()
    n_h = data.values[:, 4].tolist()
    l = data.values[:, 2].tolist()
    return p, h, n_p, n_h, l


def read_dataset(dataset_path, size=None):
    data = pd.read_csv(dataset_path, header=None)
    x = data.values[:, 0].tolist()
    y = data.values[:, 1].tolist()
    if size is not None:
        return x[:size], y[:size]
    return x, y


def read_dataset_te_negative_generation(dataset_path, size=None):
    data = pd.read_csv(dataset_path)
    p = data.values[:, 0].tolist()
    h = data.values[:, 1].tolist()
    y = data.values[:, 2].tolist()
    if size is not None:
        return p[:size], h, y[:size]
    return p, h, y


def read_dataset_te(dataset_path, size=None):
    data = pd.read_csv(dataset_path, header=None)
    data_as_list = []
    for p, h, l in data.values[1:]:
        if "rte" in dataset_path:
            data_as_list.append({"sentence1": p, "sentence2": h, "label": int(l)})
        else:
            data_as_list.append({"premise": p, "hypothesis": h, "label": int(l)})

    if size is not None:
        return data_as_list[:size]
    return data_as_list


def read_dataset_te_negation_augmented(dataset_path, size=None):
    data = pd.read_csv(dataset_path, header=None)
    data_as_list = []
    for p, h, l in data.values[1:]:
        if "rte" in dataset_path:
            if int(l) != 2:
                data_as_list.append({"sentence1": p, "sentence2": h, "label": int(l)})
        else:
            if int(l) <= 2:
                data_as_list.append({"premise": p, "hypothesis": h, "label": int(l)})

    if size is not None:
        return data_as_list[:size]
    return data_as_list


def read_dataset_te_negation_augmented_unlikelihood_loss(dataset_path, size=None):
    data = pd.read_csv(dataset_path, header=None)
    data_as_list = []
    for p, h, l in data.values[1:]:
        if "rte" in dataset_path:
            if int(l) != 2:
                data_as_list.append(
                    {
                        "sentence1": p,
                        "sentence2": h,
                        "label": int(l),
                        "known_label": True,
                    }
                )
            # else:
            #     data_as_list.append(
            #         {
            #             "sentence1": p,
            #             "sentence2": h,
            #             "label": int(l),
            #             "known_label": False,
            #         }
            #     )
        else:
            if int(l) <= 2:
                data_as_list.append(
                    {
                        "premise": p,
                        "hypothesis": h,
                        "label": int(l),
                        "known_label": True,
                    }
                )
            else:
                if int(l) == 3:
                    data_as_list.append(
                        {
                            "premise": p,
                            "hypothesis": h,
                            "label": 0,
                            "known_label": False,
                        }
                    )
                elif int(l) == 4:
                    data_as_list.append(
                        {
                            "premise": p,
                            "hypothesis": h,
                            "label": 2,
                            "known_label": False,
                        }
                    )
    if size is not None:
        return data_as_list[:size]
    return data_as_list


def mnli_to_csv():
    train_dataset_mnli = load_dataset("glue", "mnli", split=f"train")
    val_dataset_mnli = load_dataset("glue", "mnli", split=f"validation_matched")

    with open("train_mnli.csv", "w", newline="", encoding="utf8") as f:
        out = csv.writer(f)
        out.writerow(["premise", "hypothesis", "label"])
        for i in train_dataset_mnli:
            out.writerow([i["premise"], i["hypothesis"], i["label"]])

    with open("val_mnli.csv", "w", newline="", encoding="utf8") as f:
        out = csv.writer(f)
        out.writerow(["premise", "hypothesis", "label"])
        for i in val_dataset_mnli:
            out.writerow([i["premise"], i["hypothesis"], i["label"]])


def snli_to_csv():
    train_dataset_snli = load_dataset("snli", split=f"train")
    val_dataset_snli = load_dataset("snli", split=f"validation")

    with open("train_snli.csv", "w", newline="", encoding="utf8") as f:
        out = csv.writer(f)
        out.writerow(["premise", "hypothesis", "label"])
        for i in train_dataset_snli:
            out.writerow([i["premise"], i["hypothesis"], i["label"]])

    with open("val_snli.csv", "w", newline="", encoding="utf8") as f:
        out = csv.writer(f)
        out.writerow(["premise", "hypothesis", "label"])
        for i in val_dataset_snli:
            out.writerow([i["premise"], i["hypothesis"], i["label"]])


def rte_to_csv():
    train_dataset_rte = load_dataset("glue", "rte", split=f"train")
    val_dataset_rte = load_dataset("glue", "rte", split=f"validation")

    with open("train_rte.csv", "w", newline="", encoding="utf8") as f:
        out = csv.writer(f)
        out.writerow(["sentence1", "sentence2", "label"])
        for i in train_dataset_rte:
            out.writerow([i["sentence1"], i["sentence2"], i["label"]])

    with open("val_rte.csv", "w", newline="", encoding="utf8") as f:
        out = csv.writer(f)
        out.writerow(["sentence1", "sentence2", "label"])
        for i in val_dataset_rte:
            out.writerow([i["sentence1"], i["sentence2"], i["label"]])


def sample_dataset(dataset_name, split, size=None):
    if dataset_name == "rte" and split == "train":
        dataset = load_dataset("glue", "rte", split=f"train")
    elif dataset_name == "rte" and split == "val":
        dataset = load_dataset("glue", "rte", split=f"validation")
    elif dataset_name == "mnli" and split == "train":
        dataset = load_dataset("glue", "mnli", split=f"train")
    elif dataset_name == "mnli" and split == "val":
        dataset = load_dataset("glue", "mnli", split=f"validation_matched")
    elif dataset_name == "snli" and split == "train":
        dataset = load_dataset("snli", split=f"train")
    elif dataset_name == "snli" and split == "val":
        dataset = load_dataset("snli", split=f"validation")

    dataset_instances = []
    if dataset_name == "rte":
        for d in dataset:
            try:
                str_d = f"rte sentence1: {d['sentence1']} sentence2: {d['sentence2']}"
                label = d["label"]
                dataset_instances.append([str_d, ID_TO_LABEL_RTE[int(label)]])
            except Exception as e:
                pass
    elif dataset_name == "mnli" or dataset_name == "snli":
        for d in dataset:
            try:
                str_d = f"mnli hypothesis: {d['hypothesis']} premise: {d['premise']}"
                label = d["label"]
                dataset_instances.append([str_d, ID_TO_LABEL_SNLI_MNLI[int(label)]])
            except Exception as e:
                pass
    if size is not None:
        random_dataset_instances = sample(dataset_instances, size)
        return random_dataset_instances
    return dataset_instances


def extract_sentences(dataset_path, output_file_name):
    data = pd.read_csv(dataset_path)

    with open(output_file_name, "w", encoding="utf-8") as f:
        for i in data.values[:, :2]:
            f.write(str(i[0]) + "\n")
            f.write(str(i[1]) + "\n")
