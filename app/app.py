import streamlit as st
import os
from scripts.run_pipeline import process_pdf  # Assuming this runs your full PDF → Summary → Story pipeline

st.set_page_config(page_title="📚 Paper2Story", layout="wide")

st.title("📄 Paper2Story: Research Paper Simplifier")

st.markdown("""
Upload a PDF research paper and get:
- A concise **summary**
- An easy-to-understand **story**
- (Coming soon) a **Q&A chatbot** based on the paper
""")

uploaded_file = st.file_uploader("Upload a research paper (PDF)", type=["pdf"])

if uploaded_file is not None:
    with st.spinner("⏳ Processing... This may take a minute..."):
        # Save the file temporarily
        temp_path = os.path.join("data", "uploaded.pdf")
        with open(temp_path, "wb") as f:
            f.write(uploaded_file.read())

        # Run the full pipeline
        try:
            output = process_pdf(temp_path)  # Output = {'summary': ..., 'story': ...}
        except Exception as e:
            st.error(f"Error during processing: {e}")
            st.stop()

    st.success("✅ Done! Scroll down to view results.")

    st.subheader("📌 Summary")
    st.text_area("Generated Summary", value=output.get("summary", "No summary generated."), height=250)

    st.subheader("📖 Story")
    st.text_area("Readable Story", value=output.get("story", "No story generated."), height=400)

    # (Optional) Future integration
    # st.subheader("💬 Q&A Chatbot")
    # st.write("Coming soon...")

else:
    st.info("👆 Upload a PDF to get started.")
