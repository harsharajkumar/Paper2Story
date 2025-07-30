import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
import sys
import os
import torch
import pandas as pd
from datasets import Dataset
from functools import partial
import evaluate

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer,
    DataCollatorForSeq2Seq
)
from peft import get_peft_model, PromptTuningConfig, TaskType
from utils.chunking import chunk_text
from utils.eval_utils import evaluate_prompt_model  # ✅ Import here

# Set environment variable to avoid tokenizer warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_and_preprocess_data():
    df = pd.read_parquet("/Users/harsharajkumar/Downloads/youtube project/llm-story-generator/dataset.parquet")
    df["chunks"] = df["input"].apply(lambda x: chunk_text(str(x), max_chunk_size=400))
    df = df.explode("chunks").reset_index(drop=True)
    df = df[["chunks", "summary1"]].dropna()
    df = df.rename(columns={"chunks": "chunk"})

    df["chunk"] = df["chunk"].astype(str)
    df["summary1"] = df["summary1"].astype(str)
    df = df[df["chunk"].str.len() > 10]
    return Dataset.from_pandas(df)

def setup_model(model_id="google/flan-t5-small"):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

    peft_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=10,
        tokenizer_name_or_path=model_id
    )

    model = get_peft_model(base_model, peft_config)
    return model, tokenizer

def preprocess_function(batch, tokenizer):
    inputs = ["summarize: " + text for text in batch["chunk"]]

    model_inputs = tokenizer(
        inputs,
        max_length=256,
        truncation=True,
        padding="max_length"
    )

    with tokenizer.as_target_tokenizer():
        labels = tokenizer(
            batch["summary1"],
            max_length=128,
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
        output_dir="models/summary_prompt",
        per_device_train_batch_size=2,
        learning_rate=3e-4,
        num_train_epochs=5,
        logging_dir="./logs/summary",
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

def main():
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Using device: {device}")

    dataset = load_and_preprocess_data()
    model, tokenizer = setup_model()
    model.to(device)

    trainer, tokenized_dataset = setup_training(model, tokenizer, dataset)
    trainer.train()

    model_path = "models/summary_prompt"
    model.save_pretrained(model_path)
    tokenizer.save_pretrained(model_path)

    # ✅ Evaluate model
    evaluate_prompt_model(model_path, dataset)

if __name__ == "__main__":
    main()
