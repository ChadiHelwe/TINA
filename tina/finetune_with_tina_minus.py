import torch
from torch.utils.data import ConcatDataset, DataLoader
from tqdm import tqdm
from transformers.optimization import AdamW

from tina.collator.textual_entailment_collator import AutoTextualEntailmentCollator
from tina.dataset_dataloaders.textual_entailment_dataset_dataloader import (
    TextualEntailmentDataset,
    TextualEntailmentNegationAugmentedDataset,
)
from tina.model import AutoModelTE


def finetune_with_tina_minus(
    pretrained_bert_model="bert-base-cased",
    train_dataset_size=100,
    dataset_name="train_snli",
    batch_size=32,
    epochs=3,
    learning_rate=1e-3,
    weight_decay=0,
    num_labels=2,
    save_model_name="best_textual_entailment_negation_augmented_model",
    device="cpu",
    split=False,
):
    model = AutoModelTE(pretrained_bert_model, num_labels)
    model.to(device)

    te_original_dataset = TextualEntailmentDataset(
        f"data/nli/{dataset_name}.csv", train_dataset_size
    )

    te_augmented_dataset = TextualEntailmentNegationAugmentedDataset(
        f"data/nli/{dataset_name}_negation_augmented.csv", train_dataset_size
    )

    te_dataset = ConcatDataset([te_original_dataset, te_augmented_dataset])

    if "rte" in dataset_name:
        textual_entailment_collator = AutoTextualEntailmentCollator(
            pretrained_bert_model, "rte"
        )
    else:
        textual_entailment_collator = AutoTextualEntailmentCollator(
            pretrained_bert_model, "mnli"
        )

    if split:
        cnt_te_dataset = len(te_dataset)
        cnt_train_te_dataset = int(0.9 * cnt_te_dataset)

        cnt_val_te_dataset = cnt_te_dataset - cnt_train_te_dataset

        train_dataset, val_dataset = torch.utils.data.random_split(
            te_dataset, (cnt_train_te_dataset, cnt_val_te_dataset)
        )

        train_dataloader = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=textual_entailment_collator,
        )
        val_dataloader = DataLoader(
            val_dataset, batch_size=batch_size, collate_fn=textual_entailment_collator,
        )
    else:
        train_dataloader = DataLoader(
            te_dataset,
            batch_size=batch_size,
            shuffle=True,
            collate_fn=textual_entailment_collator,
        )

    optimizer = AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

    best_loss = 10000

    for epoch in range(0, epochs):
        train_loss = 0
        val_loss = 0

        print(f"Epoch: {epoch}")

        model.train()
        pbar = tqdm(train_dataloader)
        for x, y in pbar:
            loss = model(x.to(device), y=y.to(device)).loss
            pbar.set_postfix({"loss": loss.item()})
            train_loss += loss.item()
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        train_loss /= len(train_dataloader)

        print(f"Training Loss: {train_loss}")

        if split:
            model.eval()
            with torch.no_grad():
                for x, y in tqdm(val_dataloader):
                    loss = model(x.to(device), y=y.to(device)).loss
                    val_loss += loss.item()

                val_loss /= len(val_dataloader)

                print(f"Val Loss: {val_loss}")

            if best_loss > val_loss:
                print("We have a new best model")
                best_loss = val_loss
                torch.save(model, f"models/{save_model_name}.pkl")
        else:
            if best_loss > train_loss:
                print("We have a new best model")
                best_loss = train_loss
                torch.save(model, f"models/{save_model_name}.pkl")

    del model
    torch.cuda.empty_cache()
