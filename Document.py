# Import necessary libraries
import streamlit as st
from transformers import pipeline
import fitz  # PyMuPDF for PDF handling

# Load the summarization model (e.g., BERT, GPT, T5)
summarizer = pipeline("summarization")

# Streamlit app layout
st.title("Research Paper Summarizer")
st.write("Elevate your research experience with our text summarization tool!")

# User input: Text area for input
user_input = st.text_area("Enter your research paper or article:", height=200)

# PDF upload
pdf_file = st.file_uploader("Upload a PDF file", type=["pdf"])

# Summarize button
if st.button("Summarize"):
    if user_input:
        # Generate summary from text input
        summary = summarizer(user_input, max_length=150, min_length=30, do_sample=False)
        st.write("Summary:")
        st.write(summary[0]["summary_text"])
    elif pdf_file:
        # Extract text from uploaded PDF
        pdf_doc = fitz.open(stream=pdf_file.read(), filetype="pdf")
        pdf_text = ""
        for page in pdf_doc:
            pdf_text += page.get_text()

        # Generate summary from extracted PDF text
        summary = summarizer(pdf_text, max_length=150, min_length=30, do_sample=False)
        st.write("Summary from PDF:")
        st.write(summary[0]["summary_text"])
    else:
        st.warning("Please enter some text or upload a PDF to summarize.")

# Additional features (e.g., error handling) can be added as needed

# Footer
st.write("Powered by LangChain and deployed on AWS")

# Link to the deployed app
st.write("Explore the live app here")
