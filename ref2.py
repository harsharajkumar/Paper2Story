import os
import pdfplumber
import torch
import nltk
from tkinter import Tk, filedialog
from nltk.tokenize import sent_tokenize, word_tokenize
from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    TrainingArguments,
    Trainer
)
from peft import get_peft_model, LoraConfig, PromptTuningConfig, TaskType

nltk.download('punkt')

# ---------- SETUP ----------
base_model_name = "google/flan-t5-base"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)

# --------- PDF → TEXT ---------
def process_pdf(file_path):
    text = ""
    with pdfplumber.open(file_path) as pdf:
        for page in pdf.pages:
            text += page.extract_text() or ""
    return text

# --------- TEXT CHUNKING ---------
def chunk_text(text, max_tokens=500):
    sentences = sent_tokenize(text)
    chunks, current_chunk, current_len = [], [], 0
    for sent in sentences:
        sent_tokens = len(tokenizer.tokenize(sent))
        if current_len + sent_tokens <= max_tokens:
            current_chunk.append(sent)
            current_len += sent_tokens
        else:
            chunks.append(" ".join(current_chunk))
            current_chunk = [sent]
            current_len = sent_tokens
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ---------- STEP 1: PDF → LONG SUMMARY ----------
def generate_long_summary(paper_text, prompt_model):
    chunks = chunk_text(paper_text, max_tokens=500)
    summaries = []

    for chunk in chunks:
        input_text = "summarize: " + chunk
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = prompt_model.generate(**inputs, max_length=150)
        summary = tokenizer.decode(outputs[0], skip_special_tokens=True)
        summaries.append(summary)

    return " ".join(summaries)

# ---------- STEP 1.5: COMPRESS SUMMARY TO 750 WORDS ----------
def compress_summary(long_summary, prompt_model, word_limit=750):
    chunks = chunk_text(long_summary, max_tokens=500)
    compressed_chunks = []
    total_words = 0

    for chunk in chunks:
        if total_words >= word_limit:
            break

        input_text = "shorten this summary chunk: " + chunk
        inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
        outputs = prompt_model.generate(**inputs, max_length=150)
        short_piece = tokenizer.decode(outputs[0], skip_special_tokens=True)

        words = word_tokenize(short_piece)
        words_to_add = word_limit - total_words

        if len(words) > words_to_add:
            short_piece = " ".join(words[:words_to_add])

        compressed_chunks.append(short_piece)
        total_words += len(word_tokenize(short_piece))

    return " ".join(compressed_chunks)

# ---------- STEP 2: SUMMARY → STORY ----------
def generate_story(summary_text, lora_model, word_limit=750):
    input_text = "Tell a simple story based on: " + summary_text
    inputs = tokenizer(input_text, return_tensors="pt", max_length=512, truncation=True)
    outputs = lora_model.generate(**inputs, max_length=300)
    story = tokenizer.decode(outputs[0], skip_special_tokens=True)

    words = word_tokenize(story)
    if len(words) > word_limit:
        story = " ".join(words[:word_limit])

    return story

# ---------- PROMPT TUNED SUMMARY MODEL ----------
def load_prompt_model():
    prompt_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    prompt_config = PromptTuningConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        num_virtual_tokens=20,
        tokenizer_name_or_path=base_model_name
    )
    return get_peft_model(prompt_model, prompt_config)

# ---------- LORA STORY MODEL ----------
def load_lora_model():
    lora_model = AutoModelForSeq2SeqLM.from_pretrained(base_model_name)
    lora_config = LoraConfig(
        task_type=TaskType.SEQ_2_SEQ_LM,
        r=8,
        lora_alpha=16,
        lora_dropout=0.1,
        bias="none",
        target_modules=["q", "v"]
    )
    return get_peft_model(lora_model, lora_config)

# ---------- MAIN ----------
if __name__ == "__main__":
    root = Tk()
    root.withdraw()
    pdf_path = filedialog.askopenfilename(filetypes=[("PDF Files", "*.pdf")])
    if not pdf_path:
        print("No file selected.")
        exit()

    print("📄 Extracting text from PDF...")
    paper_text = process_pdf(pdf_path)

    print("🧠 Loading Prompt-tuned model for summarization...")
    prompt_model = load_prompt_model()

    print("📚 Generating long summary...")
    long_summary = generate_long_summary(paper_text, prompt_model)

    print("✂️ Compressing to 750-word summary...")
    compressed_summary = compress_summary(long_summary, prompt_model, word_limit=750)
    print("\n✅ FINAL SUMMARY (Max 750 words):\n", compressed_summary)

    print("📘 Loading LoRA model for story generation...")
    lora_model = load_lora_model()

    print("📖 Generating story from summary...")
    story = generate_story(compressed_summary, lora_model, word_limit=750)
    print("\n🎉 FINAL STORY (Max 750 words):\n", story)
