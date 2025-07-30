# run_pipeline.py

import os
import torch
import nltk
from nltk.tokenize import word_tokenize
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from peft import PeftModel, PeftConfig
from utils.chunking import chunk_text
from utils.eval_utils import compute_metrics

nltk.download("punkt")

# --------------------------------------------------
# 1. Load base summarization model (pretrained only)
# --------------------------------------------------
summary_model_id = "google/flan-t5-base"
summary_tokenizer = AutoTokenizer.from_pretrained(summary_model_id)
summary_model = AutoModelForSeq2SeqLM.from_pretrained(summary_model_id)
summary_model.eval()

# ---------------------------------------------------
# 2. Load fine-tuned story generation model with LoRA
# ---------------------------------------------------
lora_model_path = "models/story_lora"
lora_config = PeftConfig.from_pretrained(lora_model_path)

story_base_model = AutoModelForSeq2SeqLM.from_pretrained(lora_config.base_model_name_or_path)
story_tokenizer = AutoTokenizer.from_pretrained(lora_config.base_model_name_or_path)
story_model = PeftModel.from_pretrained(story_base_model, lora_model_path)
story_model.eval()

# ----------------------------
# 3. Set Device
# ----------------------------
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
summary_model.to(device)
story_model.to(device)


# ---------------------------------------------
# 4. Stage 1: Generate long summary from chunks
# ---------------------------------------------
def generate_long_summary(text, chunk_size=500, max_output_len=150):
    chunks = chunk_text(text, tokenizer=summary_tokenizer, max_tokens=chunk_size)
    all_summaries = []

    for chunk in chunks:
        input_text = "summarize: " + chunk
        inputs = summary_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
        outputs = summary_model.generate(**inputs, max_length=max_output_len)
        summary = summary_tokenizer.decode(outputs[0], skip_special_tokens=True)
        all_summaries.append(summary)

    return " ".join(all_summaries)


# --------------------------------------------
# 5. Stage 1.5: Compress summary to ~750 words
# --------------------------------------------
def compress_summary(long_summary, word_limit=750):
    chunks = chunk_text(long_summary, tokenizer=summary_tokenizer, max_tokens=400)
    compressed_chunks = []
    total_words = 0

    for chunk in chunks:
        if total_words >= word_limit:
            break
        input_text = "summarize: " + chunk
        inputs = summary_tokenizer(input_text, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        outputs = summary_model.generate(**inputs, max_length=128)
        summary = summary_tokenizer.decode(outputs[0], skip_special_tokens=True)

        words = word_tokenize(summary)
        if total_words + len(words) > word_limit:
            words = words[:word_limit - total_words]
        total_words += len(words)
        compressed_chunks.append(" ".join(words))

    return " ".join(compressed_chunks)


# -------------------------------------------------------
# 6. Stage 2: Generate story using fine-tuned LoRA model
# -------------------------------------------------------
def generate_story(summary_text, word_limit=750):
    input_text = "Write a simple story based on: " + summary_text
    inputs = story_tokenizer(input_text, return_tensors="pt", truncation=True, padding=True, max_length=512).to(device)
    outputs = story_model.generate(**inputs, max_length=300)
    story = story_tokenizer.decode(outputs[0], skip_special_tokens=True)

    words = word_tokenize(story)
    if len(words) > word_limit:
        story = " ".join(words[:word_limit])
    return story


# ----------------------------
# 7. MAIN
# ----------------------------
if __name__ == "__main__":
    # Replace this with actual text from PDF
    input_text = """
    Large Language Models (LLMs) like GPT and T5 have revolutionized NLP tasks.
    These models use transformer architectures and are pre-trained on diverse datasets...
    """

    print("📚 Generating long summary...")
    long_summary = generate_long_summary(input_text)

    print("\n✂️ Compressing to 750-word summary...")
    short_summary = compress_summary(long_summary, word_limit=750)
    print("\n✅ Final Summary:\n", short_summary)

    print("\n📖 Generating story using LoRA model...")
    story = generate_story(short_summary)
    print("\n🎉 Final Story:\n", story)

    # Optional: Evaluation (if target/ground truth is available)
    # metrics = compute_metrics(pred=story, target=target_story)
    # print("📊 Evaluation:", metrics)
