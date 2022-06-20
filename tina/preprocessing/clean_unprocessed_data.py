import numpy as np
import pandas as pd
from transformers import T5Tokenizer


def merge_datasets():
    tokenizer = T5Tokenizer.from_pretrained("t5-3b")
    data = pd.read_csv(
        "data/unprocessed_data/out.tsv",
        sep="\t",
        header=None,
        encoding="cp1252",
        on_bad_lines="skip",
    )
    data = data.dropna()

    data1 = pd.read_csv(
        "data/unprocessed_data/out_npi.tsv",
        sep="\t",
        header=None,
        encoding="cp1252",
        on_bad_lines="skip",
    )
    data1 = data1.dropna()

    data2 = pd.concat([data, data1]).drop_duplicates().reset_index(drop=True)
    data2 = data2.sample(frac=1).reset_index(drop=True)
    data2 = data2[data2[0].apply(lambda row: len(tokenizer(row).input_ids) < 120)]

    msk = np.random.rand(len(data2)) < 0.9
    data2_train = data2[msk]
    data2_val = data2[~msk]

    data2_1 = data2_train[[0, 2, 1]]
    data2_2 = data2_train[[0, 2, 1]]

    data2_2.loc[:, [0, 2, 1]] = data2_2.loc[:, [2, 0, 1]].values

    print(data2_1)
    print(data2_2)

    data3 = pd.concat([data2_1, data2_2]).reset_index(drop=True)
    data3 = data3.sample(frac=1).reset_index(drop=True)

    data3.to_csv("train.csv", header=False, index=False)

    data2_1 = data2_val[[0, 2, 1]]
    data2_2 = data2_val[[0, 2, 1]]

    data2_2.loc[:, [0, 2, 1]] = data2_2.loc[:, [2, 0, 1]].values

    print(data2_1)
    print(data2_2)

    data3 = pd.concat([data2_1, data2_2]).reset_index(drop=True)
    data3 = data3.sample(frac=1).reset_index(drop=True)

    data3.to_csv("val.csv", header=False, index=False)


def analyze_dataset(dataset_path):
    data = pd.read_csv(dataset_path, header=None)
    print(data)
    print(data[2])
    print(data[2].value_counts())


def extract_sentences(dataset_path, output_file_name):
    data = pd.read_csv(dataset_path)

    with open(output_file_name, "w", encoding="utf-8") as f:
        for i in data.values[:, :2]:
            f.write(str(i[0]) + "\n")
            f.write(str(i[1]) + "\n")
