# Paper2Story 📚✍️

**Paper2Story** is an AI pipeline that simplifies academic research papers into:
- Concise summaries
- Human-readable story format

🔍 **What it does**
1. 📄 Extracts & chunks text from scientific PDFs
2. ✨ Summarizes the chunks using a prompt-tuned model
3. 📝 Converts summaries into engaging narratives using LoRA-fine-tuned LLMs

🛠️ **Tech Stack**
- HuggingFace Transformers
- PEFT (LoRA)
- PyTorch
- Streamlit (for UI)

📂 **Repo Structure**
- `scripts/`: Training & inference scripts
- `utils/`: Chunking, evaluation, and data utils
- `models/`: Saved model weights
- `app/`: Streamlit frontend
- `data/`: Input/output parquet files

🚀 **Coming Soon**
- QA chatbot from story
- End-to-end UI

---

Made with ❤️ by Harsha Rajkumar
