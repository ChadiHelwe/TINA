import csv

import torch
import torch.nn.functional as f
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import AutoModelForSequenceClassification

from tina.collator.grammar_collator import GrammarCollator
from tina.dataset_dataloaders.grammar_dataset_dataloader import GrammarDataset

device = "cpu"


dataset_name = "clean_train_mnli_negation_with_npi.csv"
new_dataset_name = "clean_train_mnli_negation_with_npi_grammar_checked.csv"

model = AutoModelForSequenceClassification.from_pretrained(
    "textattack/distilbert-base-cased-CoLA",
)
model.to(device)

dataset = GrammarDataset(f"data/nli/{dataset_name}")
collator = GrammarCollator()
dataloader = DataLoader(dataset, batch_size=32, collate_fn=collator,)

with open(f"{new_dataset_name}", "w", newline="", encoding="utf-8") as fl:
    out = csv.writer(fl)
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
    with torch.no_grad():
        for t_ps, t_hs, ps, hs, n_ps, n_hs, ls in tqdm(dataloader):

            ps_logits = model(**t_ps.to(device)).logits
            ps_values = f.softmax(ps_logits, 1)
            hs_logits = model(**t_hs.to(device)).logits
            hs_values = f.softmax(hs_logits, 1)

            for p_v, h_v, p, h, n_p, n_h, l in zip(
                ps_values, hs_values, ps, hs, n_ps, n_hs, ls
            ):
                if p_v[1].item() > 0.6 and h_v[1].item() > 0.6:
                    out.writerow([p, h, l, "-", "-", n_p, n_h])
