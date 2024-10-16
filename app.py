import os
import json
import logging
from datetime import datetime
from pymongo import MongoClient
from transformers import pipeline, BartForConditionalGeneration, BartTokenizer
from PyPDF2 import PdfReader
from sklearn.feature_extraction.text import TfidfVectorizer
import streamlit as st
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# Initialize logging
logging.basicConfig(level=logging.INFO)

# MongoDB setup
# client = MongoClient('mongodb://localhost:27017/') change uri as per use
db = client['pdf_database']
collection = db['pdf_documents']

# Load the model for summarization
model_name = "facebook/bart-large-cnn"
model = BartForConditionalGeneration.from_pretrained(model_name)
tokenizer = BartTokenizer.from_pretrained(model_name)
summarizer = pipeline("summarization", model=model, tokenizer=tokenizer)

def summarize_large_text(text):
    # Adjust chunk size based on the text length
    if len(text) > 3000:
        chunk_size = 1024  
    elif len(text) > 1000:
        chunk_size = 512   
    else:
        chunk_size = 256   

    if len(text) > chunk_size:
        chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
        final_summary = ""
        for chunk in chunks:
            try:
                summary = summarizer(chunk, max_length=450, min_length=40, do_sample=False)
                if summary and isinstance(summary, list) and 'summary_text' in summary[0]:
                    final_summary += summary[0]['summary_text'] + " "
                else:
                    logging.error("Summarizer response does not contain 'summary_text'.")
            except Exception as e:
                logging.error(f"Error generating summary: {e}")
        return final_summary.strip()
    else:
        try:
            summary = summarizer(text, max_length=450, min_length=40, do_sample=False)
            if summary and isinstance(summary, list) and 'summary_text' in summary[0]:
                return summary[0]['summary_text']
            else:
                logging.error("Summarizer response format is unexpected.")
                return ""
        except Exception as e:
            logging.error(f"Error generating summary: {e}")
            return ""

def extract_keywords(text, num_keywords=5):
    vectorizer = TfidfVectorizer(stop_words='english', max_features=num_keywords)
    X = vectorizer.fit_transform([text])
    keywords = vectorizer.get_feature_names_out()
    return keywords.tolist()

def process_pdf(pdf_file):
    try:
        pdfreader = PdfReader(pdf_file)
        text = ''
        for page in pdfreader.pages:
            content = page.extract_text()
            if content:
                text += content
        if not text:
            logging.warning("No text extracted from the PDF. Skipping this file.")
            return None
        
        document_info = {
            'document_name': pdf_file.name,
            'size': pdf_file.size,
            'created_at': datetime.now(),
            'summary': None,
            'keywords': None,
        }

        # Insert metadata into MongoDB
        doc_id = collection.insert_one(document_info).inserted_id

        # Summarization and keyword extraction
        summary = summarize_large_text(text)
        keywords = extract_keywords(text)

        # Update MongoDB with summary and keywords
        collection.update_one(
            {'_id': doc_id},
            {'$set': {'summary': summary, 'keywords': keywords}}
        )

        return summary, keywords

    except Exception as e:
        logging.error(f"Error processing PDF: {e}")
        return None

# Streamlit UI
st.title("PDF Summary and Keyword Extractor")

# Concurrency settings
max_workers = 4  # You can adjust this based on your machine's capability
executor = ThreadPoolExecutor(max_workers=max_workers)

pdf_files = st.file_uploader("Upload PDF files", type='pdf', accept_multiple_files=True)

if pdf_files:
    st.write("Processing your PDFs...")
    start_time = time.time()

    futures = {executor.submit(process_pdf, pdf): pdf.name for pdf in pdf_files}
    
    for future in as_completed(futures):
        pdf_name = futures[future]
        try:
            summary, keywords = future.result()
            if summary and keywords:
                st.subheader(f"Summary for {pdf_name}")
                st.write(summary)
                st.subheader("Keywords")
                st.write(", ".join(keywords))
            else:
                st.write(f"An error occurred while processing {pdf_name}.")
        except Exception as e:
            st.write(f"Error processing {pdf_name}: {e}")

    end_time = time.time()
    elapsed_time = end_time - start_time
    st.write(f"Total time taken to process documents: {elapsed_time:.2f} seconds.")
