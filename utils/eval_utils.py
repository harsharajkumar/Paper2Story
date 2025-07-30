from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from peft import PeftModel, PeftConfig
from rouge_score import rouge_scorer
from statistics import mean
import torch


def compute_rouge(predictions, references):
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    scores = [scorer.score(ref, pred) for pred, ref in zip(predictions, references)]

    avg_scores = {
        "rouge1": mean([score["rouge1"].fmeasure for score in scores]),
        "rouge2": mean([score["rouge2"].fmeasure for score in scores]),
        "rougeL": mean([score["rougeL"].fmeasure for score in scores]),
    }
    return avg_scores


def evaluate_prompt_model(model_path, dataset, num_samples=100):
    device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"

    # Load base model and tokenizer
    config = PeftConfig.from_pretrained(model_path)
    base_model = AutoModelForSeq2SeqLM.from_pretrained(config.base_model_name_or_path)
    model = PeftModel.from_pretrained(base_model, model_path).to(device)
    tokenizer = AutoTokenizer.from_pretrained(config.base_model_name_or_path)

    # Prepare test samples
    test_samples = dataset.select(range(min(num_samples, len(dataset))))
    inputs = ["summarize: " + x for x in test_samples["chunk"]]
    inputs_tokenized = tokenizer(inputs, return_tensors="pt", padding=True, truncation=True).to(device)

    # Generate outputs
    outputs = model.generate(**inputs_tokenized, max_new_tokens=128)
    decoded_preds = tokenizer.batch_decode(outputs, skip_special_tokens=True)
    decoded_labels = [label for label in test_samples["summary1"]]

    # Evaluate
    rouge_scores = compute_rouge(decoded_preds, decoded_labels)
    print("ROUGE scores:", rouge_scores)
    return rouge_scores
