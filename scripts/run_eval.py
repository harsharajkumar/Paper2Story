# run_eval.py
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
import pandas as pd
from datasets import Dataset
from utils.chunking import chunk_text
from utils.eval_utils import evaluate_prompt_model

# Avoid tokenizer parallelism warning
os.environ["TOKENIZERS_PARALLELISM"] = "false"

def load_data():
    df = pd.read_parquet("./dataset.parquet")  
    df["chunks"] = df["input"].apply(lambda x: chunk_text(str(x), max_chunk_size=400))
    df = df.explode("chunks").reset_index(drop=True)
    df = df[["chunks", "summary1"]].dropna()
    df = df.rename(columns={"chunks": "chunk"})
    df["chunk"] = df["chunk"].astype(str)
    df["summary1"] = df["summary1"].astype(str)
    df = df[df["chunk"].str.len() > 10]
    return Dataset.from_pandas(df)

def main():
    dataset = load_data()
    model_path = "models/summary_prompt"
    evaluate_prompt_model(model_path, dataset)

if __name__ == "__main__":
    main()
