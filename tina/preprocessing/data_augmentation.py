import csv

import pandas as pd

from tina.utils import (
    ID_TO_LABEL_RTE,
    ID_TO_LABEL_SNLI_MNLI,
    LABEL_TO_ID_RTE,
    LABEL_TO_ID_SNLI_MNLI,
)


def data_augmentation(path_dataset, path_new_dataset, rte_task):
    dataset = pd.read_csv(path_dataset).iterrows()

    with open(path_new_dataset, "w", newline="", encoding="utf-8") as f:
        out = csv.writer(f)
        if rte_task:
            out.writerow(["sentence1", "sentence2", "label"])
        else:
            out.writerow(["premise", "hypothesis", "label"])

        for _, i in dataset:
            premise = i["Premise"]
            hypothesis = i["Hypothesis"]
            negated_premise = i["Negated Premise"]
            negated_hypothesis = i["Negated Hypothesis"]

            if rte_task:
                label = ID_TO_LABEL_RTE[i["Label"]]
            else:
                label = ID_TO_LABEL_SNLI_MNLI[i["Label"]]

            if label == "entailment":
                if rte_task:
                    out.writerow(
                        [premise, negated_hypothesis, LABEL_TO_ID_RTE["not_entailment"]]
                    )
                    out.writerow(
                        [negated_premise, hypothesis, LABEL_TO_ID_RTE["not_entailment"]]
                    )
                    out.writerow(
                        [
                            negated_hypothesis,
                            negated_premise,
                            LABEL_TO_ID_RTE["entailment"],
                        ]
                    )
                    out.writerow(
                        [negated_hypothesis, premise, LABEL_TO_ID_RTE["not_entailment"]]
                    )
                else:
                    out.writerow(
                        [
                            premise,
                            negated_hypothesis,
                            LABEL_TO_ID_SNLI_MNLI["contradiction"],
                        ]
                    )
                    out.writerow(
                        [
                            negated_premise,
                            hypothesis,
                            LABEL_TO_ID_SNLI_MNLI["not_entailment"],
                        ]
                    )
                    out.writerow(
                        [
                            negated_hypothesis,
                            negated_premise,
                            LABEL_TO_ID_SNLI_MNLI["entailment"],
                        ]
                    )
                    out.writerow(
                        [
                            negated_hypothesis,
                            premise,
                            LABEL_TO_ID_SNLI_MNLI["contradiction"],
                        ]
                    )

            elif label == "contradiction":
                if not rte_task:
                    out.writerow(
                        [
                            premise,
                            negated_hypothesis,
                            LABEL_TO_ID_SNLI_MNLI["entailment"],
                        ]
                    )
                    out.writerow(
                        [
                            negated_premise,
                            hypothesis,
                            LABEL_TO_ID_SNLI_MNLI["not_contradiction"],
                        ]
                    )
                    out.writerow(
                        [
                            hypothesis,
                            negated_premise,
                            LABEL_TO_ID_SNLI_MNLI["entailment"],
                        ]
                    )
                    out.writerow(
                        [hypothesis, premise, LABEL_TO_ID_SNLI_MNLI["contradiction"]]
                    )

            elif label == "neutral":
                if not rte_task:
                    out.writerow(
                        [premise, negated_hypothesis, LABEL_TO_ID_SNLI_MNLI["neutral"]]
                    )
            elif label == "not_entailment":
                if rte_task:
                    out.writerow(
                        [
                            premise,
                            negated_hypothesis,
                            LABEL_TO_ID_RTE["not_contradiction"],
                        ]
                    )
                    out.writerow(
                        [
                            negated_hypothesis,
                            negated_premise,
                            LABEL_TO_ID_RTE["not_entailment"],
                        ]
                    )
