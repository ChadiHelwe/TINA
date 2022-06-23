from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as f
from transformers import (
    AutoModelForSequenceClassification,
    AutoTokenizer,
    T5ForConditionalGeneration,
    T5Tokenizer,
)

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


class T5(nn.Module):
    def __init__(self, pretrained_t5_model: str) -> None:
        super().__init__()
        self.model = T5ForConditionalGeneration.from_pretrained(pretrained_t5_model)
        self.tokenizer = T5Tokenizer.from_pretrained(pretrained_t5_model)

    def forward(self, x, y=None):
        """
        If the label is not None, then return the model with the input and the label. Otherwise, return
        the model with the input and the decoder_input_ids

        :param x: a dictionary of input tensors
        :param y: labels
        :return: The model is being returned.
        """
        if y is not None:
            return self.model(**x, labels=y)
        return self.model(**x, decoder_input_ids=x["input_ids"])

    def predict_te(
        self, phrase_1: str, phrase_2: str, prompt: str, device: str = "cpu"
    ) -> str:
        """
        The function takes in two sentences, a prompt, and a device (cpu or gpu) and returns the prediction

        :param phrase_1: The first sentence in the pair
        :type phrase_1: str
        :param phrase_2: The hypothesis or the second sentence in the pair
        :type phrase_2: str
        :param prompt: :param prompt: the prompt to use for the generation. Can be either "rte" or "snli"
        :type prompt: str
        :param device: str = "cpu", defaults to cpu
        :type device: str (optional)
        :return: The output of the model.
        """
        with torch.no_grad():
            if prompt == "rte":
                input_ids = self.tokenizer(
                    f"{prompt} sentence1: {phrase_1} sentence2: {phrase_2}",
                    return_tensors="pt",
                ).input_ids
            else:
                input_ids = self.tokenizer(
                    f"{prompt} hypothesis: {phrase_2} premise: {phrase_1}",
                    return_tensors="pt",
                ).input_ids
            outputs = self.model.generate(
                input_ids.to(device),
                num_beams=10,
                num_return_sequences=1,
                no_repeat_ngram_size=1,
                remove_invalid_values=True,
            )
            pred_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return pred_output

    def predict_te_greedy(
        self, phrase_1: str, phrase_2: str, prompt: str, device: str = "cpu"
    ) -> str:
        """
        The function takes in two sentences, a prompt, and a device (cpu or gpu) and returns the prediction

        :param phrase_1: The first sentence in the pair
        :type phrase_1: str
        :param phrase_2: The hypothesis or the premise
        :type phrase_2: str
        :param prompt: the prompt to use for the generation. Can be either "rte" or "snli"
        :type prompt: str
        :param device: The device to run the model on, defaults to cpu
        :type device: str (optional)
        :return: The output of the model.
        """
        with torch.no_grad():
            if prompt == "rte":
                input_ids = self.tokenizer(
                    f"{prompt} sentence1: {phrase_1} sentence2: {phrase_2}",
                    return_tensors="pt",
                ).input_ids
            else:
                input_ids = self.tokenizer(
                    f"{prompt} hypothesis: {phrase_2} premise: {phrase_1}",
                    return_tensors="pt",
                ).input_ids
            outputs = self.model.generate(input_ids.to(device), num_return_sequences=1)
            pred_output = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return pred_output


class AutoModelTE(nn.Module):
    def __init__(self, pretrained_bert_model: str, num_labels: int) -> None:
        super().__init__()
        self.model = AutoModelForSequenceClassification.from_pretrained(
            pretrained_bert_model, num_labels=num_labels
        )
        self.tokenizer = AutoTokenizer.from_pretrained(pretrained_bert_model)
        if pretrained_bert_model == "gpt2":
            self.tokenizer.add_special_tokens(SPECIAL_TOKENS)
            self.model.resize_token_embeddings(len(self.tokenizer))
            self.model.config.pad_token_id = self.model.config.eos_token_id

    def forward(self, x, y=None):
        if y is not None:
            return self.model(**x, labels=y)
        return self.model(**x, decoder_input_ids=x["input_ids"])

    def predict_te(
        self, phrase_1: str, phrase_2: str, prompt: str, device: str = "cpu"
    ) -> str:
        """
        The function takes in two sentences, a prompt, and a device (cpu or gpu) and returns the prediction

        :param phrase_1: The first sentence in the pair
        :type phrase_1: str
        :param phrase_2: The hypothesis or the premise
        :type phrase_2: str
        :param prompt: the prompt to use for the generation. Can be either "rte" or "snli"
        :type prompt: str
        :param device: The device to run the model on, defaults to cpu
        :type device: str (optional)
        :return: The output of the model.
        """

        with torch.no_grad():
            inputs = self.tokenizer(
                phrase_1, phrase_2, padding=True, return_tensors="pt"
            )
            logits = self.model(**inputs.to(device)).logits
            predicted_output = logits.argmax().item()
        if prompt == "rte":
            return ID_TO_LABEL_RTE[predicted_output]

        return ID_TO_LABEL_SNLI_MNLI[predicted_output]

    def predict(
        self, phrase_1: str, phrase_2: str, prompt: str, device: str = "cpu"
    ) -> str:
        """
        The function takes in two phrases, a prompt, and a device (cpu or gpu) and returns the predicted
        output

        :param phrase_1: The first phrase in the pair
        :type phrase_1: str
        :param phrase_2: The second phrase in the prompt
        :type phrase_2: str
        :param prompt: the prompt to use for the generation. Can be either "rte" or "snli"
        :type prompt: str
        :param device: str = "cpu", defaults to cpu
        :type device: str (optional)
        :return: The predicted output
        """
        with torch.no_grad():
            inputs = self.tokenizer(
                phrase_1, phrase_2, padding=True, return_tensors="pt"
            )
            logits = self.model(**inputs.to(device)).logits
            predicted_output = f.softmax(logits)

        return predicted_output
