import csv

import pandas as pd
import spacy
from nltk.tokenize import word_tokenize
from sklearn import datasets
from tqdm import tqdm


def reformat_negation_dataset(
    path_negated_data, path_non_negated_data, output_data, task
):
    negated_data = pd.read_csv(
        path_negated_data, sep="\t", encoding="utf-8", header=None
    )
    data = pd.read_csv(path_non_negated_data, encoding="utf-8")

    negated_data_iter = negated_data.iterrows()
    data = data.iterrows()

    if task == "rte":
        premise_key = "sentence1"
        hypothesis_key = "sentence2"
    else:
        premise_key = "premise"
        hypothesis = "hypothesis"

    with open(output_data, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(
            ["Premise", "Hypothesis", "Label", "Negated Premise", "Negated Hypothesis",]
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
                        row[premise_key],
                        row[hypothesis_key],
                        row["label"],
                        neg_row1[2],
                        neg_row2[2],
                    ]
                )
                cnt += 1

        print(cnt)


def check_negation_dataset(path_training_data, path_testing_data, clean_dataset_name):
    training_data = pd.read_csv(path_training_data, encoding="utf-8")
    testing_data = pd.read_csv(path_testing_data, sep="\t", encoding="cp1252")

    # nlp = spacy.load("en_core_web_lg")
    cnt_inst = 0
    cnt_no_inst = 0
    with open(clean_dataset_name, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        out.writerow(
            ["Premise", "Hypothesis", "Label", "Negated Premise", "Negated Hypothesis",]
        )
        for i in tqdm(training_data.values[:, :]):
            training_neg_prem = set(word_tokenize(i[3]))
            training_neg_hyp = set(word_tokenize(i[4]))
            find_sim = False
            for j in testing_data.values[:, :]:
                testing_neg_prem = set(word_tokenize(j[1]))
                testing_neg_hyp = set(word_tokenize(j[2]))
                # sim_prem = training_neg_prem.similarity(testing_neg_prem)
                # sim_hyp = training_neg_hyp.similarity(testing_neg_hyp)
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
                    find_sim = True
                    cnt_no_inst += 1
                    break
            if not find_sim:
                cnt_inst += 1
                out.writerow(i)
        print(cnt_inst)
        print(cnt_no_inst)
