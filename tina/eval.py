import csv

import pandas as pd
import torch
from sklearn.metrics import accuracy_score

from tina.model import T5
from tina.utils import LABEL_TO_ID_RTE, LABEL_TO_ID_SNLI_MNLI


def evaluate_per_negation_dataset(
    path_model,
    dataset_name,
    results_file_name,
    device="cpu",
    greedy=False,
    pretrained_model=False,
):
    if pretrained_model:
        model = T5(path_model)
    else:
        model = torch.load(f"models/{path_model}.pkl")
    model.to(device)

    model.eval()
    with open(
        f"results/{results_file_name}_{dataset_name}.csv",
        "w",
        newline="",
        encoding="utf8",
    ) as out_f, open(
        f"results/predictions_{results_file_name}_{dataset_name}.csv",
        "w",
        newline="",
        encoding="utf8",
    ) as out_pred:
        out = csv.writer(out_f)
        out_p = csv.writer(out_pred)

        out.writerow(["Model", "Dataset", "Accuracy"])
        out_p.writerow(["Premise", "Hypothesis", "True Value", "Predicted Value"])

        data = pd.read_csv(
            f"data/negated_nli/{dataset_name}.txt", "\t", encoding="unicode_escape"
        )

        true_values = []
        pred_values = []

        if dataset_name == "RTE":
            prompt = "rte"
        else:
            prompt = "mnli"
        cnt = 0
        for d in data.values:
            try:
                if dataset_name == "RTE":
                    if greedy:
                        pred_value = LABEL_TO_ID_RTE[
                            model.predict_te_greedy(d[1], d[2], prompt, device)
                        ]
                        pred_values.append(pred_value)
                    else:
                        pred_value = LABEL_TO_ID_RTE[
                            model.predict_te(d[1], d[2], prompt, device)
                        ]
                        pred_values.append(pred_value)

                    true_value = LABEL_TO_ID_RTE[d[3]]
                    true_values.append(true_value)

                    out_p.writerow([d[1], d[2], true_value, pred_value])
                    cnt += 1
                else:
                    if greedy:
                        pred_value = LABEL_TO_ID_SNLI_MNLI[
                            model.predict_te_greedy(d[1], d[2], prompt, device)
                        ]
                        pred_values.append(pred_value)
                    else:
                        pred_value = LABEL_TO_ID_SNLI_MNLI[
                            model.predict_te(d[1], d[2], prompt, device)
                        ]
                        pred_values.append(pred_value)

                    true_value = LABEL_TO_ID_SNLI_MNLI[d[3]]
                    true_values.append(true_value)

                    out_p.writerow([d[1], d[2], true_value, pred_value])
                    cnt += 1
            except Exception as err:
                print(err)

        print(f"Total {cnt}")
        if len(pred_values) > 0:
            acc_score = accuracy_score(true_values, pred_values)
            out.writerow([f"{path_model}", f"{dataset_name}", acc_score])
        else:
            out.writerow([f"{path_model}", f"{dataset_name}", 0])


def evaluate_per_dataset(
    path_model,
    dataset_name,
    results_file_name,
    device="cpu",
    greedy=False,
    pretrained_model=False,
):

    if pretrained_model:
        model = T5(path_model)
    else:
        model = torch.load(f"models/{path_model}.pkl")
    model.to(device)

    model.eval()
    with open(
        f"results/{results_file_name}_{dataset_name}.csv",
        "w",
        newline="",
        encoding="utf8",
    ) as out_f, open(
        f"results/predictions_{results_file_name}_{dataset_name}.csv",
        "w",
        newline="",
        encoding="utf8",
    ) as out_pred:
        out = csv.writer(out_f)
        out_p = csv.writer(out_pred)

        out.writerow(["Model", "Dataset", "Accuracy"])
        out_p.writerow(["Premise", "Hypothesis", "True Value", "Predicted Value"])

        data = pd.read_csv(f"data/nli/{dataset_name}.csv", encoding="unicode_escape")

        true_values = []
        pred_values = []

        if "rte" in dataset_name:
            prompt = "rte"
        else:
            prompt = "mnli"
        cnt = 0
        for d in data.values:
            try:
                if "rte" in dataset_name:
                    if greedy:
                        pred_value = LABEL_TO_ID_RTE[
                            model.predict_te_greedy(d[0], d[1], prompt, device)
                        ]
                        pred_values.append(pred_value)
                    else:
                        pred_value = LABEL_TO_ID_RTE[
                            model.predict_te(d[0], d[1], prompt, device)
                        ]
                        pred_values.append(pred_value)

                    true_value = d[2]
                    true_values.append(true_value)

                    out_p.writerow([d[0], d[1], true_value, pred_value])
                    cnt += 1
                else:
                    if greedy:
                        pred_value = LABEL_TO_ID_SNLI_MNLI[
                            model.predict_te_greedy(d[0], d[1], prompt, device)
                        ]
                        pred_values.append(pred_value)
                    else:
                        pred_value = LABEL_TO_ID_SNLI_MNLI[
                            model.predict_te(d[0], d[1], prompt, device)
                        ]
                        pred_values.append(pred_value)

                    true_value = d[2]
                    true_values.append(true_value)

                    out_p.writerow([d[0], d[1], true_value, pred_value])
                    cnt += 1
            except Exception as err:
                print(err)
        print(f"Total: {cnt}")
        if len(pred_values) > 0:
            acc_score = accuracy_score(true_values, pred_values)
            out.writerow([f"{path_model}", f"{dataset_name}", acc_score])
        else:
            out.writerow([f"{path_model}", f"{dataset_name}", 0])
