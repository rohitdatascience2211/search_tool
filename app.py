import streamlit as st
import pandas as pd
import numpy as np
import torch
from transformers import BertTokenizer, BertModel
import faiss
import os

# Set the environment variable to avoid OpenMP conflicts
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Load BERT model and tokenizer only once
@st.cache_resource  # Use Streamlit's caching for resources like models
def load_model():
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    model = BertModel.from_pretrained('bert-base-uncased')
    model.eval()
    return tokenizer, model

# Function to get BERT embeddings
def get_bert_embeddings(text, tokenizer, model):
    inputs = tokenizer(text, return_tensors='pt', truncation=True, padding=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    embeddings = outputs.last_hidden_state[:, 0, :]
    return embeddings.squeeze().numpy()

# Search function with FAISS index
def search_courses(query, df, tokenizer, model, top_k=5, similarity_threshold=70.0):
    # Generate embeddings if not already done
    course_embeddings = np.array([get_bert_embeddings(text, tokenizer, model) for text in df['combined_text']]).astype('float32')
    
    index = faiss.IndexFlatL2(course_embeddings.shape[1])  # L2 distance
    index.add(course_embeddings)
    
    query_embedding = get_bert_embeddings(query, tokenizer, model).astype('float32')
    D, I = index.search(np.array([query_embedding]), top_k)
    
    results = []
    for j, i in enumerate(I[0]):
        if D[0][j] < similarity_threshold:
            continue
        course = df.iloc[i]
        results.append((course['S.NO'], course['TITLE'], course['DESCRIPTION'], D[0][j]))
    results = sorted(results, key=lambda x: x[3], reverse=True)
    
    return results

# Streamlit UI
st.title("Course Recommendation System")
st.write("Search for courses related to your interests.")

# Load model and tokenizer
tokenizer, model = load_model()

# Load course data
@st.cache_data
def load_data():
    df = pd.read_csv("cleaned_courses.csv")
    df['combined_text'] = df['TITLE'] + " " + df['DESCRIPTION']
    return df

df = load_data()

# User input
query = st.text_input("Enter a topic or keyword:", "Learn about machine learning")
similarity_threshold = st.slider("Set similarity threshold:", 0, 100, 40)
top_k = st.slider("Select number of results to display:", 1, 10, 5)

# Perform search and display results
if st.button("Search"):
    with st.spinner("Searching for relevant courses..."):
        results = search_courses(query, df, tokenizer, model, top_k=top_k, similarity_threshold=similarity_threshold)
    if results:
        for course_id, title, description, score in results:
            st.write(f"**Course ID**: {course_id}")
            st.write(f"**Title**: {title}")
            st.write(f"**Description**: {description}")
            st.write(f"**Similarity score**: {score:.2f}")
            st.write("---")
    else:
        st.write("No relevant courses found.")
