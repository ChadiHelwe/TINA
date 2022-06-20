import torch
from transformers import AutoTokenizer, T5TokenizerFast

from tina.utils import ID_TO_LABEL_RTE, ID_TO_LABEL_SNLI_MNLI

SPECIAL_TOKENS = {
    "bos_token": "<|endoftext|>",
    "eos_token": "<|endoftext|>",
    "pad_token": "[PAD]",
    "additional_special_tokens": [
        "[SYS]",
        "[USR]",
        "[KG]",
        "[SUB]",
        "[PRED]",
        "[OBJ]",
        "[TRIPLE]",
        "[SEP]",
        "[Q]",
        "[DOM]",
    ],
}


class T5TextualEntaillmentCollator:
    def __init__(self, pretrained_t5_tokenizer, prompt):
        self.tokenizer = T5TokenizerFast.from_pretrained(pretrained_t5_tokenizer)
        self.prompt = prompt

    def __call__(self, batch):
        """
        It takes a batch of data, and returns a batch of data that is tokenized and padded
        
        :param batch: a batch of data from the dataset
        :return: The input_ids of the labels
        """
        premises_and_hypotheses = []
        labels = []
        for b in batch:
            try:
                if self.prompt == "rte":
                    premises_and_hypotheses.append(
                        f"{self.prompt} sentence1: {b['sentence1']} sentence2: {b['sentence2']}"
                    )
                    labels.append(f"{ID_TO_LABEL_RTE[b['label']]}")
                else:
                    premises_and_hypotheses.append(
                        f"{self.prompt} hypothesis: {b['hypothesis']} premise: {b['premise']}"
                    )
                    if b["label"] in ID_TO_LABEL_SNLI_MNLI:
                        labels.append(f"{ID_TO_LABEL_SNLI_MNLI[b['label']]}")
                    else:
                        labels.append("neutral")
            except Exception as e:
                print(e.args)

        batch_x = self.tokenizer(
            premises_and_hypotheses, padding=True, return_tensors="pt"
        )
        batch_y = self.tokenizer(labels, padding=True, return_tensors="pt")

        return batch_x, batch_y["input_ids"]


class T5TextualEntaillmentUnlikelihoodLossCollator:
    def __init__(self, pretrained_t5_tokenizer, prompt):
        self.tokenizer = T5TokenizerFast.from_pretrained(pretrained_t5_tokenizer)
        self.prompt = prompt

    def __call__(self, batch):
        """
        It takes a batch of data, tokenizes it, and returns the tokenized data, the labels, and the
        known/unknown labels.
        
        :param batch: a list of dictionaries, each dictionary contains the premise, hypothesis, label,
        and known_label
        :return: The batch_x is a dictionary with keys: input_ids, attention_mask, token_type_ids.
        """
        premises_and_hypotheses = []
        labels = []
        known_labels = []
        unknown_labels = []
        for b in batch:
            try:
                if self.prompt == "rte":
                    premises_and_hypotheses.append(
                        f"{self.prompt} sentence1: {b['sentence1']} sentence2: {b['sentence2']}"
                    )
                    labels.append(f"{ID_TO_LABEL_RTE[b['label']]}")

                    if b["known_label"]:
                        known_labels.append(1)
                        unknown_labels.append(0)
                    else:
                        known_labels.append(0)
                        unknown_labels.append(1)

                else:
                    premises_and_hypotheses.append(
                        f"{self.prompt} hypothesis: {b['hypothesis']} premise: {b['premise']}"
                    )
                    if b["label"] in ID_TO_LABEL_SNLI_MNLI:
                        labels.append(f"{ID_TO_LABEL_SNLI_MNLI[b['label']]}")
                    else:
                        labels.append("neutral")

                    if b["known_label"]:
                        known_labels.append(1)
                        unknown_labels.append(0)
                    else:
                        known_labels.append(0)
                        unknown_labels.append(1)

            except Exception as e:
                print(e.args)

        batch_x = self.tokenizer(
            premises_and_hypotheses, padding=True, return_tensors="pt"
        )
        batch_y = self.tokenizer(labels, padding=True, return_tensors="pt")
        batch_known_labels = torch.tensor(known_labels)
        batch_unknown_labels = torch.tensor(unknown_labels)

        return batch_x, batch_y["input_ids"], batch_known_labels, batch_unknown_labels


class AutoTextualEntailmentCollator:
    def __init__(self, pretrained_bert_tokenizer, prompt) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_tokenizer)
        self.prompt = prompt
        if pretrained_bert_tokenizer == "gpt2":
            self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

    def __call__(self, batch):
        """
        It takes a batch of data, and returns a batch of data that is tokenized and padded
        
        :param batch: a batch of data from the dataset
        :return: The input_ids of the labels
        """
        premises = []
        hypotheses = []
        labels = []

        for b in batch:
            try:
                if self.prompt == "rte":
                    premises.append(str(b["sentence1"]))
                    hypotheses.append(str(b["sentence2"]))
                    labels.append(int(b["label"]))
                else:
                    premises.append(str(b["premise"]))
                    hypotheses.append(str(b["hypothesis"]))
                    if int(b["label"]) in ID_TO_LABEL_SNLI_MNLI:
                        labels.append(int(b["label"]))
                    else:
                        labels.append(1)
            except Exception as e:
                print(e.args)

        batch_x = self.tokenizer(
            premises, hypotheses, padding=True, return_tensors="pt"
        )
        batch_y = torch.tensor(labels)

        return batch_x, batch_y


class AutoTextualEntailmentUnlikelihoodLossCollator:
    def __init__(self, pretrained_bert_tokenizer, prompt) -> None:
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_tokenizer)
        self.prompt = prompt
        if pretrained_bert_tokenizer == "gpt2":
            self.tokenizer.add_special_tokens(SPECIAL_TOKENS)

    def __call__(self, batch):
        """
        It takes a batch of data, tokenizes it, and returns the tokenized data, the labels, and the
        known/unknown labels.
        
        :param batch: a list of dictionaries, each dictionary contains the premise, hypothesis, label,
        and known_label
        :return: The batch_x is a dictionary with keys: input_ids, attention_mask, token_type_ids.
        """
        premises = []
        hypotheses = []
        labels = []
        known_labels = []
        unknown_labels = []

        for b in batch:
            try:
                if self.prompt == "rte":
                    premises.append(str(b["sentence1"]))
                    hypotheses.append(str(b["sentence2"]))
                    labels.append(int(b["label"]))
                    if b["known_label"]:
                        known_labels.append(1)
                        unknown_labels.append(0)
                    else:
                        known_labels.append(0)
                        unknown_labels.append(1)
                else:
                    premises.append(str(b["premise"]))
                    hypotheses.append(str(b["hypothesis"]))
                    if int(b["label"]) in ID_TO_LABEL_SNLI_MNLI:
                        labels.append(int(b["label"]))
                    else:
                        labels.append(1)

                    if b["known_label"]:
                        known_labels.append(1)
                        unknown_labels.append(0)
                    else:
                        known_labels.append(0)
                        unknown_labels.append(1)

            except Exception as e:
                print(e.args)

        batch_x = self.tokenizer(
            premises, hypotheses, padding=True, return_tensors="pt"
        )
        batch_y = torch.tensor(labels)
        batch_known_labels = torch.tensor(known_labels)
        batch_unknown_labels = torch.tensor(unknown_labels)

        return batch_x, batch_y, batch_known_labels, batch_unknown_labels
