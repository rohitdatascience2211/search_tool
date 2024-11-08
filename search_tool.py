import numpy as np
import pandas as pd
import torch
from sentence_transformers import SentenceTransformer, util
import os

os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

model = SentenceTransformer('all-MiniLM-L6-v2')
def get_sbert_embeddings(text):
    return model.encode(text, convert_to_tensor=True)

def search_courses(query, df, top_k=5):

    course_embeddings = [get_sbert_embeddings(text) for text in df['combined_text']]
    
    query_embedding = get_sbert_embeddings(query)

    cosine_similarities = util.pytorch_cos_sim(query_embedding, torch.stack(course_embeddings))[0]
    results = [
        (df.iloc[i]['S.NO'], df.iloc[i]['TITLE'], df.iloc[i]['DESCRIPTION'], score.item()) 
        for i, score in enumerate(cosine_similarities)
    ]

    sorted_results = sorted(results, key=lambda x: x[3], reverse=True)

    unique_results = []
    seen_ids = set()
    
    for course in sorted_results:
        course_id = course[0]
        if course_id not in seen_ids:
            unique_results.append(course)
            seen_ids.add(course_id)
        if len(unique_results) == top_k:
            break
    
    return unique_results

if __name__ == "__main__":
    df = pd.read_csv("cleaned_courses.csv")
    df['combined_text'] = (df['TITLE'] + " " + df['DESCRIPTION']).str.lower().str.strip()
    
    query = input("Enter a topic or keyword to search for courses: ")
    results = search_courses(query, df)

    if results:
        print(f"Results for query: '{query}'")
        for course_id, title, description, score in results:
            print(f"Course ID: {course_id}")
            print(f"Title: {title}")
            print(f"Description: {description}")
            print(f"Similarity score: {score:.4f}")
            print('---')
    else:
        print("No relevant courses found.")
