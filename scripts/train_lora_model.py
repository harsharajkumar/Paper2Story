
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from datasets import Dataset
from functools import partial

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, LoraConfig, TaskType
from utils.chunking import chunk_text
from utils.eval_utils import compute_rouge

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_and_preprocess_data(path="./dataset.parquet"):
    df = pd.read_parquet(path)
    df = df[["summary1", "story"]].dropna()

    # Optional chunking
    df["chunks"] = df["summary1"].apply(lambda x: chunk_text(x))
    df = df.explode("chunks").reset_index(drop=True)
    df = df.rename(columns={"chunks": "summary_chunk"})
    df["summary_chunk"] = df["summary_chunk"].astype(str)
    df["story"] = df["story"].astype(str)

    return Dataset.from_pandas(df)

def setup_model(model_id="google/flan-t5-base"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    peft_config = LoraConfig(
        r=8,
        lora_alpha=16,
        target_modules=["q", "v"],
        lora_dropout=0.1,
        bias="none",
        task_type=TaskType.SEQ_2_SEQ_LM
    )

    model = get_peft_model(base_model, peft_config)
    return model, tokenizer

def preprocess_function(batch, tokenizer):
    inputs = ["Tell a story based on: " + x for x in batch["summary_chunk"]]

    model_inputs = tokenizer(
        inputs,
        max_length=512,
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["story"],
            max_length=256,
            truncation=True,
            padding="max_length"
        )

    model_inputs["labels"] = labels["input_ids"]
    return model_inputs

def setup_training(model, tokenizer, dataset):
    tokenized_dataset = dataset.map(
        partial(preprocess_function, tokenizer=tokenizer),
        batched=True,
        remove_columns=dataset.column_names
    )

    args = TrainingArguments(
        output_dir="models/lora_story",
        per_device_train_batch_size=2,
        learning_rate=3e-4,
        num_train_epochs=5,
        logging_dir="./logs/story",
        logging_steps=50,
        save_strategy="epoch",
        eval_strategy="epoch",
        report_to="none",
        optim="adafactor",
        remove_unused_columns=False,
        fp16=torch.cuda.is_available(),
        gradient_accumulation_steps=4 if torch.backends.mps.is_available() else 1
    )

    trainer = Trainer(
        model=model,
        args=args,
        train_dataset=tokenized_dataset,
        eval_dataset=tokenized_dataset.select(range(100)),
        tokenizer=tokenizer,
        data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
    )

    return trainer, tokenized_dataset

def evaluate_model(trainer, tokenizer, dataset):
    predictions = trainer.predict(dataset.select(range(100))).predictions
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    decoded_labels = [example['story'] for example in dataset.select(range(100))]
    rouge_scores = compute_rouge(decoded_preds, decoded_labels)
    print("ROUGE scores:", rouge_scores)

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = load_and_preprocess_data()
    model, tokenizer = setup_model()
    model.to(device)

    trainer, tokenized_dataset = setup_training(model, tokenizer, dataset)
    trainer.train()

    model_path = "models/story_lora"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    evaluate_model(trainer, tokenizer, dataset)

if __name__ == "__main__":
    main()
