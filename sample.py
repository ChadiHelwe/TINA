import pandas as pd


data = pd.read_csv("data/nli/clean_train_rte_negation_with_npi_grammar_checked.csv")
df = pd.DataFrame(
    data, columns=["Premise", "Hypothesis", "Negated Premise", "Negated Hypothesis"]
)
df.sample(30, ignore_index=True).to_csv("sample_30_rte.csv")
