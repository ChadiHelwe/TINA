# TINA: Textual Inference with Negation Augmentation

## Installation

Clone this repository

```bash
git clone https://github.com/ChadiHelwe/FailBERT.git
pip install -r requirements.txt
```

## Reproducing Augmented Datasets
### Extract Sentences
```bash
python cli.py --extract_sentences --input_1 data/nli/train_rte.csv --output extracted_train_rte.txt
```

### Format to ConLL-U
```bash
python cli.py --convert_to_conllu --input_1 train_extracted_rte.txt --output conllu_train_rte.txt
```

### Negate
```bash
python cli.py --negate --input_1 conllu_train_rte.txt --output negated_train_rte.tsv
```


### Reformat Negated Dataset
```bash
python cli.py --reformat_negation_dataset --input_1 negated_train_rte.tsv --input_2 data/nli/train_rte.csv --output negated_formated_train_rte.csv --task rte
```

### Check if there are Testing Negated Instances in Training Negated Instances
```bash
python cli.py --check_negation_dataset --input_1 negated_formated_train_rte.csv --input_2 data/negated_nli/RTE.txt --output checked_negated_formated_train_rte.csv
```


### Check Grammar
```bash
python cli.py --check_grammar --input_1 checked_negated_formated_train_rte.csv  --output grammar_checked_negated_formated_train_rte.csv --device cpu
```

### Data Augmentation
```bash
python cli.py --data_augmentation --input_1 grammar_checked_negated_formated_train_rte.csv --output train_rte_negation_augmented.csv --task rte
```

## Finetuning Models

### Without Splitting 
```bash
python cli.py --finetune --model bert-base-cased  --task rte --learning_rate 1e-4 --epochs 10 --weight_decay 0 --batch_size 8  --runs 2 --device cpu
```

### With Splitting 
```bash
python cli.py --finetune --model bert-base-cased  --task rte --learning_rate 1e-4 --epochs 10 --weight_decay 0 --batch_size 8  --runs 2 --device cpu --split
```

## Finetuning Models with TINA Minus

## Without Splitting 
```bash
python cli.py --finetune_with_tina_minus --model bert-base-cased  --task rte --learning_rate 1e-4 --epochs 10 --weight_decay 0 --batch_size 8  --runs 2 --device cpu
```

## With Splitting 
```bash
python cli.py --finetune_with_tina_minus --model bert-base-cased  --task rte --learning_rate 1e-4 --epochs 10 --weight_decay 0 --batch_size 8 --runs 2 --device cpu --split
```

## Finetuning Models with TINA

## Without Splitting 
```bash
python cli.py --finetune_with_tina --model bert-base-cased  --task rte --learning_rate 1e-4 --epochs 10 --weight_decay 0 --batch_size 8 --runs 2 --device cpu
```

## With Splitting 
```bash
python cli.py --finetune_with_tina --model bert-base-cased  --task rte --learning_rate 1e-4 --epochs 10 --weight_decay 0 --batch_size 8 --runs 2 --device cpu --split
```

