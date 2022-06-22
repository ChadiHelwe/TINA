import argparse
import sys

import torch

from tina.eval import evaluate_per_dataset, evaluate_per_negation_dataset
from tina.finetune import finetune
from tina.finetune_with_tina import finetune_with_tina
from tina.finetune_with_tina_minus import finetune_with_tina_minus
from tina.preprocessing.check_grammar_correct import check_grammar
from tina.preprocessing.data_augmentation import data_augmentation
from tina.preprocessing.reformat_negation_dataset import (
    check_negation_dataset,
    reformat_negation_dataset,
)

PRETRAINED_MODEL = "bert-base-cased"
TRAIN_DATASET_SIZE = 10
VAL_DATASET_SIZE = 10
BATCH_SIZE = 32
EPOCHS = 12
LEARNING_RATE = 1e-4
DEVICE = "cpu"


def finetune_experiments(
    pretrained_model,
    task,
    learning_rate,
    epochs,
    weight_decay,
    batch_size,
    split,
    runs,
    device,
):
    print(f"Finetune {pretrained_model}")
    try:
        if task == "snli":
            train_dataset = "train_snli"
            test_dataset = "val_snli"
            neg_dataset = "SNLI"
            num_labels = 3
        elif task == "mnli":
            train_dataset = "train_mnli"
            test_dataset = "val_mnli"
            neg_dataset = "MNLI"
            num_labels = 3
        elif task == "rte":
            train_dataset = "train_rte"
            test_dataset = "val_rte"
            neg_dataset = "RTE"
            num_labels = 2
        else:
            raise Exception("You can use only one of these tasks: snli, mnli, rte")

        for exp_cnt in range(0, runs):
            torch.manual_seed(exp_cnt)
            print(f"Dataset {neg_dataset} Experiment {exp_cnt}")
            name_best_model = f"best_finetuned_model_{pretrained_model}_{task}_{exp_cnt}".replace(
                "/", ""
            )
            finetune(
                pretrained_model,
                TRAIN_DATASET_SIZE,
                train_dataset,
                batch_size,
                epochs,
                learning_rate,
                weight_decay,
                num_labels,
                name_best_model,
                device,
                split,
            )
            evaluate_per_dataset(
                f"{name_best_model}",
                test_dataset,
                f"results_{name_best_model}",
                device=device,
                pretrained_model=False,
            )

            evaluate_per_negation_dataset(
                f"{name_best_model}",
                neg_dataset,
                f"results_{name_best_model}_neg",
                device=device,
                pretrained_model=False,
            )
    except Exception as e:
        print(e.args[0])


def finetune_with_tina_minus_experiments(
    pretrained_model,
    task,
    learning_rate,
    epochs,
    weight_decay,
    batch_size,
    split,
    runs,
    device,
):
    print(f"Finetune {pretrained_model} with TINA Minus")
    try:
        if task == "snli":
            train_dataset = "train_snli"
            test_dataset = "val_snli"
            neg_dataset = "SNLI"
            num_labels = 3
        elif task == "mnli":
            train_dataset = "train_mnli"
            test_dataset = "val_mnli"
            neg_dataset = "MNLI"
            num_labels = 3
        elif task == "rte":
            train_dataset = "train_rte"
            test_dataset = "val_rte"
            neg_dataset = "RTE"
            num_labels = 2
        else:
            raise Exception("You can use only one of these tasks: snli, mnli, rte")

        for exp_cnt in range(0, runs):
            torch.manual_seed(exp_cnt)
            print(f"Dataset {neg_dataset} Experiment {exp_cnt}")
            name_best_model = f"best_finetuned_model_with_tina_minus_{pretrained_model}_{task}_{exp_cnt}".replace(
                "/", ""
            )
            finetune_with_tina_minus(
                pretrained_model,
                TRAIN_DATASET_SIZE,
                train_dataset,
                batch_size,
                epochs,
                learning_rate,
                weight_decay,
                num_labels,
                name_best_model,
                device,
                split,
            )

            evaluate_per_dataset(
                f"{name_best_model}",
                test_dataset,
                f"results_{name_best_model}",
                device=device,
                pretrained_model=False,
            )

            evaluate_per_negation_dataset(
                f"{name_best_model}",
                neg_dataset,
                f"results_{name_best_model}_neg",
                device=device,
                pretrained_model=False,
            )

    except Exception as e:
        print(e.args[0])


def finetune_with_tina_experiments(
    pretrained_model,
    task,
    learning_rate,
    epochs,
    weight_decay,
    batch_size,
    split,
    runs,
    device,
):
    print(f"Finetune {pretrained_model} with TINA")
    try:
        if task == "snli":
            train_dataset = "train_snli"
            test_dataset = "val_snli"
            neg_dataset = "SNLI"
            num_labels = 3
        elif task == "mnli":
            train_dataset = "train_mnli"
            test_dataset = "val_mnli"
            neg_dataset = "MNLI"
            num_labels = 3
        elif task == "rte":
            train_dataset = "train_rte"
            test_dataset = "val_rte"
            neg_dataset = "RTE"
            num_labels = 2
        else:
            raise Exception("You can use only one of these tasks: snli, mnli, rte")

        for exp_cnt in range(0, runs):
            torch.manual_seed(exp_cnt)
            print(f"Dataset {neg_dataset} Experiment {exp_cnt}")
            name_best_model = f"best_finetuned_model_with_tina_{pretrained_model}_{task}_{exp_cnt}".replace(
                "/", ""
            )
            finetune_with_tina(
                pretrained_model,
                TRAIN_DATASET_SIZE,
                train_dataset,
                batch_size,
                epochs,
                learning_rate,
                weight_decay,
                num_labels,
                name_best_model,
                device,
                split,
            )

            evaluate_per_dataset(
                f"{name_best_model}",
                test_dataset,
                f"results_{name_best_model}",
                device=device,
                pretrained_model=False,
            )

            evaluate_per_negation_dataset(
                f"{name_best_model}",
                neg_dataset,
                f"results_{name_best_model}_neg",
                device=device,
                pretrained_model=False,
            )

    except Exception as e:
        print(e.args[0])


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    my_parser.add_argument(
        "--model",
        help="transformer-based models",
        default="bert-base-uncased",
        type=str,
    )
    my_parser.add_argument("--epochs", help="Number of Epochs", default="3", type=int)
    my_parser.add_argument(
        "--learning_rate", help="Learning Rate", default=1e-5, type=float
    )
    my_parser.add_argument("--weight_decay", help="Weight Decay", default=0, type=float)
    my_parser.add_argument("--batch_size", help="Batch Size", default=32, type=int)
    my_parser.add_argument("--task", help="Task", default="snli", type=str)
    my_parser.add_argument("--runs", help="Number of Runs", default=3, type=int)
    my_parser.add_argument(
        "--split", help="Split 90/10 Training Dataset", action="store_true"
    )
    my_parser.add_argument("--device", help="Device", default="cpu", type=str)
    my_parser.add_argument("--input_1", help="Input File", type=str)
    my_parser.add_argument("--input_2", help="Input File", type=str)
    my_parser.add_argument("--output", help="Output File", type=str)

    my_parser.add_argument(
        "--finetune", help="finetune experiments", action="store_true",
    )
    my_parser.add_argument(
        "--finetune_with_tina_minus",
        help="finetune with TINA Minus experiments",
        action="store_true",
    )
    my_parser.add_argument(
        "--finetune_with_tina",
        help="finetune with TINA experiments",
        action="store_true",
    )
    my_parser.add_argument("--check_grammar", help="Check Grammar", action="store_true")
    my_parser.add_argument(
        "--data_augmentation", help="Data Augmentation", action="store_true"
    )
    my_parser.add_argument(
        "--check_negation_dataset", help="Check Negation Dataset", action="store_true"
    )
    my_parser.add_argument(
        "--reformat_negation_dataset",
        help="Reformat Negation Dataset",
        action="store_true",
    )
    try:
        args = my_parser.parse_args()
    except:
        my_parser.print_help()
        sys.exit(0)

    if args.split:
        split = True
    else:
        split = False
    if args.finetune:
        finetune_experiments(
            args.model,
            args.task,
            args.learning_rate,
            args.epochs,
            args.weight_decay,
            args.batch_size,
            split,
            args.runs,
            args.device,
        )
    elif args.finetune_with_tina_minus:
        finetune_with_tina_minus_experiments(
            args.model,
            args.task,
            args.learning_rate,
            args.epochs,
            args.weight_decay,
            args.batch_size,
            split,
            args.runs,
            args.device,
        )
    elif args.finetune_with_tina:
        finetune_with_tina_experiments(
            args.model,
            args.task,
            args.learning_rate,
            args.epochs,
            args.weight_decay,
            args.batch_size,
            split,
            args.runs,
            args.device,
        )
    elif args.check_grammar:
        check_grammar(args.input_1, args.output, args.device)
    elif args.data_augmentation:
        if args.task == "rte":
            data_augmentation(args.input_1, args.output, True)
        else:
            data_augmentation(args.input_1, args.output, False)
    elif args.check_negation_dataset:
        check_negation_dataset(args.input_1, args.input_2, args.output)
    elif args.reformat_negation_dataset:
        reformat_negation_dataset(args.input_1, args.input_2, args.output)
