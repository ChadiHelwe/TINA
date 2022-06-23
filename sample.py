import pandas as pd


data = pd.read_csv("data/nli/clean_train_mnli_negation_with_npi_grammar_checked.csv")
df = pd.DataFrame(
    data, columns=["Premise", "Hypothesis", "Negated Premise", "Negated Hypothesis"]
)
df.sample(100, ignore_index=True).to_csv("sample_100_mnli.csv")
