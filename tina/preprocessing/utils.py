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
    """
    The function takes in a json instance, and returns two sentences, one positive and one negative,
    with the object word replaced with the object word in the json instance
    
    :param json_instance: the json instance
    :param pos_statmt_key: the key in the json file that contains the positive statement
    :param neg_statmt_key: the key in the json file that contains the negative statement
    :param object_key: the key in the json file that contains the object word
    :param repl_keyword: the keyword to be replaced by the object word
    :return: A tuple of two strings.
    """
    object_word = json_instance[object_key]
    pos_statmt = json_instance[pos_statmt_key][0].replace(repl_keyword, object_word)
    neg_statmt = json_instance[neg_statmt_key][0].replace(repl_keyword, object_word)
    return pos_statmt, neg_statmt


def read_dataset_grammar(dataset_path):
    """
    It reads the dataset and returns the lists of premises, hypotheses, negative premises, negative
    hypotheses, and labels
    
    :param dataset_path: The path to the dataset file
    :return: the values of the columns in the dataset.
    """
    data = pd.read_csv(dataset_path)
    p = data.values[:, 0].tolist()
    h = data.values[:, 1].tolist()
    n_p = data.values[:, 5].tolist()
    n_h = data.values[:, 6].tolist()
    l = data.values[:, 2].tolist()
    return p, h, n_p, n_h, l


def read_dataset(dataset_path, size=None):
    """
    It reads a CSV file and returns the first two columns as lists
    
    :param dataset_path: The path to the dataset
    :param size: The number of samples to read from the dataset
    :return: the first size number of values from the x and y lists.
    """
    data = pd.read_csv(dataset_path, header=None)
    x = data.values[:, 0].tolist()
    y = data.values[:, 1].tolist()
    if size is not None:
        return x[:size], y[:size]
    return x, y


def read_dataset_te_negative_generation(dataset_path, size=None):
    """
    It reads the dataset and returns the premise, hypothesis and label
    
    :param dataset_path: The path to the dataset
    :param size: the number of samples to be read from the dataset
    :return: the premise, hypothesis and label.
    """
    data = pd.read_csv(dataset_path)
    p = data.values[:, 0].tolist()
    h = data.values[:, 1].tolist()
    y = data.values[:, 2].tolist()
    if size is not None:
        return p[:size], h, y[:size]
    return p, h, y


def read_dataset_te(dataset_path, size=None):
    """
    It reads the dataset and returns a list of dictionaries, where each dictionary contains the premise,
    hypothesis and label
    
    :param dataset_path: The path to the dataset
    :param size: the number of examples to read from the dataset. If None, all examples are read
    :return: A list of dictionaries.
    """
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
    """
    It reads the dataset, and returns a list of dictionaries, where each dictionary contains the
    premise, hypothesis, and label
    
    :param dataset_path: the path to the dataset file
    :param size: the number of examples to read from the dataset. If None, all examples are read
    :return: A list of dictionaries.
    """
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
    """
    If the label is 2, then we don't add it to the list. If the label is 3 or 4, then we add it to the
    list with a label of 0 or 2 respectively
    
    :param dataset_path: The path to the dataset
    :param size: the number of samples to read from the dataset. If None, all samples are read
    :return: A list of dictionaries.
    """
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
    """
    It takes the MNLI dataset and converts it into a CSV file
    """
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
    """
    It takes the SNLI dataset and converts it into a CSV file
    """
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
    """
    It takes the RTE dataset from the GLUE dataset and converts it into a CSV file.
    """
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
    """
    It takes in a dataset name and a split (train or val) and returns a list of tuples of the form
    (sentence, label)
    
    :param dataset_name: The name of the dataset. Can be "rte", "mnli", or "snli"
    :param split: train or val
    :param size: the number of samples you want to take from the dataset
    :return: A list of lists. Each list contains a string and a label.
    """
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


def merge_datasets(path_negated_data, path_non_negated_data, output_data):
    negated_data = pd.read_csv(
        path_negated_data, sep="\t", encoding="cp1252", header=None
    )
    data = pd.read_csv(path_non_negated_data, encoding="utf-8")

    negated_data_iter = negated_data.iterrows()
    data = data.iterrows()

    with open(output_data, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(
            [
                "Premise",
                "Hypothesis",
                "Label",
                "Backtranslated Premise",
                "Backtranslated Hypothesis",
                "Negated Premise",
                "Negated Hypothesis",
            ]
        )
        cnt = 0
        for (_, row), (_, neg_row1), (_, neg_row2) in zip(
            data, negated_data_iter, negated_data_iter
        ):
            if (
                neg_row1[4] == True
                and neg_row2[4] == True
                # and ("NPI" not in neg_row1[1])
                # and ("NPI" not in neg_row2[1])
            ):
                out.writerow(
                    [
                        row["Premise"],
                        row["Hypothesis"],
                        row["Label"],
                        row["Backtranslated Premise"],
                        row["Backtranslated Hypothesis"],
                        neg_row1[2],
                        neg_row2[2],
                    ]
                )
                cnt += 1

        print(cnt)


def check_dataset(path_training_data, path_testing_data, clean_dataset_name):
    training_data = pd.read_csv(path_training_data, encoding="utf-8")
    testing_data = pd.read_csv(path_testing_data, sep="\t", encoding="cp1252")

    cnt_inst = 0
    cnt_no_inst = 0
    with open(clean_dataset_name, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(
            [
                "Premise",
                "Hypothesis",
                "Label",
                "Backtranslated Premise",
                "Backtranslated Hypothesis",
                "Negated Premise",
                "Negated Hypothesis",
            ]
        )
        for i in tqdm(training_data.values[:, :]):
            training_neg_prem = set(word_tokenize(i[5]))
            training_neg_hyp = set(word_tokenize(i[6]))
            find_sim = False
            for j in testing_data.values[:, :]:
                testing_neg_prem = set(word_tokenize(j[1]))
                testing_neg_hyp = set(word_tokenize(j[2]))

                l_training_neg_prem = []
                l_training_neg_hyp = []
                l_testing_neg_prem = []
                l_testing_neg_hyp = []

                premise_vector = training_neg_prem.union(testing_neg_prem)
                hypothesis_vector = training_neg_hyp.union(testing_neg_hyp)

                for w in premise_vector:
                    if w in training_neg_prem:
                        l_training_neg_prem.append(1)  # create a vector
                    else:
                        l_training_neg_prem.append(0)
                    if w in testing_neg_prem:
                        l_testing_neg_prem.append(1)
                    else:
                        l_testing_neg_prem.append(0)
                c = 0

                # cosine formula
                for p in range(len(premise_vector)):
                    c += l_training_neg_prem[p] * l_testing_neg_prem[p]
                cosine_prem = c / float(
                    (sum(l_training_neg_prem) * sum(l_testing_neg_prem)) ** 0.5
                )

                for w in hypothesis_vector:
                    if w in training_neg_hyp:
                        l_training_neg_hyp.append(1)  # create a vector
                    else:
                        l_training_neg_hyp.append(0)
                    if w in testing_neg_hyp:
                        l_testing_neg_hyp.append(1)
                    else:
                        l_testing_neg_hyp.append(0)
                c = 0

                # cosine formula
                for h in range(len(hypothesis_vector)):
                    c += l_training_neg_hyp[h] * l_testing_neg_hyp[h]
                cosine_hyp = c / float(
                    (sum(l_training_neg_hyp) * sum(l_testing_neg_hyp)) ** 0.5
                )

                if cosine_prem > 0.85 and cosine_hyp > 0.85:
                    print(i[5], i[6])
                    print(j[1], j[2])
                    find_sim = True
                    cnt_no_inst += 1
                    break
            if not find_sim:
                cnt_inst += 1
                out.writerow(i)
        print(cnt_inst)
        print(cnt_no_inst)
