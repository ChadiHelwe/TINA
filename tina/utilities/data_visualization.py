import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

sns.set_context("paper")

bert = {
    "datasets": [
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
    ],
    "accuracy": [49.10, 65.21, 58.30, 52.66, 69.50, 79.93, 69.19, 69.42, 79.93],
    "model": [
        "BERT",
        "BERT",
        "BERT",
        "BERT + TINA$^-$",
        "BERT + TINA$^-$",
        "BERT + TINA$^-$",
        "BERT + TINA",
        "BERT + TINA",
        "BERT + TINA",
    ],
}
roberta = {
    "datasets": [
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
    ],
    "accuracy": [54.46, 66.93, 74.35, 55.35, 68.55, 81.53, 67.51, 68.97, 81.53],
    "model": [
        "RoBERTa",
        "RoBERTa",
        "RoBERTa",
        "RoBERTa + TINA$^-$",
        "RoBERTa + TINA$^-$",
        "RoBERTa + TINA$^-$",
        "RoBERTa + TINA",
        "RoBERTa + TINA",
        "RoBERTa + TINA",
    ],
}
xlnet = {
    "datasets": [
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
    ],
    "accuracy": [53.77, 67.06, 68.08, 56.08, 70.09, 74.73, 66.57, 70.86, 74.73],
    "model": [
        "XLNet",
        "XLNet",
        "XLNet",
        "XLNet + TINA$^-$",
        "XLNet + TINA$^-$",
        "XLNet + TINA$^-$",
        "XLNet + TINA",
        "XLNet + TINA",
        "XLNet + TINA",
    ],
}
bart = {
    "datasets": [
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
        "SNLI$_{Neg}$",
        "MNLI$_{Neg}$",
        "RTE$_{Neg}$",
    ],
    "accuracy": [53.17, 66.60, 60.30, 52.57, 69.64, 77.33, 70.77, 70.26, 77.33],
    "model": [
        "BART",
        "BART",
        "BART",
        "BART + TINA$^-$",
        "BART + TINA$^-$",
        "BART + TINA$^-$",
        "BART + TINA",
        "BART + TINA",
        "BART + TINA",
    ],
}
# albert = {
#     "datasets": ["SNLI Neg", "MNLI Neg", "RTE Neg"],
#     "accuracy_f": [50.55, 63.70, 63.30],
#     "accuracy_f_n": [54.66, 67.59, 74.91],
#     "accuracy_f_n_u": [63.46, 66.24, 74.91],
# }

data = [[bert, roberta, xlnet], [bart, None, None]]
model_name = [["BERT", "RoBERTa", "XLNet"], ["BART", None, None]]

f, ax = plt.subplots(2, 3, sharex=False, figsize=(10, 7))
ax[1][2].set_visible(False)
ax[1][1].set_visible(False)
for i in range(0, 2):
    for j in range(0, 3):
        if data[i][j] is not None:
            df = pd.DataFrame.from_dict(data=data[i][j])
            sns.set_color_codes("colorblind")
            g = sns.barplot(
                ax=ax[i, j],
                x="datasets",
                y="accuracy",
                hue="model",
                data=df,
                palette="Greys_r",
            )
            g.legend().set_title(None)
            ax[i, j].legend(
                ncol=1, loc="upper left", frameon=False, fancybox=False, fontsize=11
            )
            ax[i, j].set_ylabel("Accuracy", fontsize=11)
            ax[i, j].set_xlabel("", fontsize=11)
            ax[i, j].set_ylim((0, 100))
            ax[i, j].tick_params(labelsize=12)
            sns.despine(left=True, bottom=True)
            ax[i, j].set_title(f"{model_name[i][j]}", fontsize=13)
f.tight_layout()

plt.rcParams.update({"text.usetex": True, "font.family": "Helvetica"})
plt.savefig("results.pdf", format="pdf", dpi=1200)
