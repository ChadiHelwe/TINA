from transformers import AutoTokenizer


class GrammarCollator:
    def __init__(self) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(
            "textattack/distilbert-base-cased-CoLA"
        )

    def __call__(self, batch):
        premises = []
        hypotheses = []
        negated_premises = []
        negated_hypotheses = []
        labels = []

        for p, h, n_p, n_h, l in batch:
            premises.append(p)
            hypotheses.append(h)
            negated_premises.append(n_p)
            negated_hypotheses.append(n_h)
            labels.append(l)

        batch_p = self.tokenizer(premises, padding=True, return_tensors="pt")
        batch_h = self.tokenizer(hypotheses, padding=True, return_tensors="pt")

        return (
            batch_p,
            batch_h,
            premises,
            hypotheses,
            negated_premises,
            negated_hypotheses,
            labels,
        )
