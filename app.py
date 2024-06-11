import requests
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import streamlit as st

# Jina AI Configuration
url = 'https://api.jina.ai/v1/embeddings'
headers = {
    'Content-Type': 'application/json',
    'Authorization': 'Bearer jina_1e1110633ba04cfba8cbc0a51c18355eYd_AjtgRWPt9a_uDXvHaJXBYOyiP'  # Replace with your actual key
}
model = 'jina-embeddings-v2-base-en'

class RAGSystem:
    def __init__(self):
        self.knowledge_base = []

    def add_document(self, text):
        embedding = self._get_embedding(text)
        self.knowledge_base.append((text, embedding))

    def _get_embedding(self, text):
        data = {
            'input': text,
            'model': model,
            'encoding_type': 'float'
        }
        response = requests.post(url, headers=headers, json=data)
        response.raise_for_status()  # Check for API errors
        embedding = np.array(response.json()['data'][0]['embedding'])
        return embedding

    def get_similar_documents(self, query, top_k=3):
        query_embedding = self._get_embedding(query)
        similarities = cosine_similarity([query_embedding], [doc[1] for doc in self.knowledge_base])[0]

        # Filter based on keywords
        filtered_indices = [
            i for i, doc in enumerate(self.knowledge_base) if query.lower() in doc[0].lower()
        ]

        if filtered_indices:
            filtered_similarities = [similarities[i] for i in filtered_indices]
            # Sort documents based on similarity in descending order
            top_indices = np.argsort(filtered_similarities)[-top_k:][::-1].astype(int)
            return [self.knowledge_base[filtered_indices[i]][0] for i in top_indices]
        else:
            return []  # Return an empty list if no relevant documents are found


# Streamlit Application
st.title("Retrieval-Augmented Generation System")

# File upload interface
uploaded_file = st.file_uploader("Upload a CSV file containing documents with a 'data' column", type=["csv"])

rag_system = RAGSystem()

if uploaded_file is not None:
    try:
        df = pd.read_csv(uploaded_file)
        if "data" in df.columns:
            # Limit to the first 100 rows
            text_data = df["data"].tolist()[:30]
            if len(text_data) < 100:
                st.warning(f"Only {len(text_data)} rows have been processed. The maximum limit is 30 rows.")
        else:
            st.error("CSV file must contain a 'data' column")
            text_data = []

    except Exception as e:
        st.error(f"An error occurred while reading the CSV file: {str(e)}")
        text_data = []

    for doc in text_data:
        rag_system.add_document(doc)

    st.success("Documents successfully added to the RAG system.")

# Query interface
query = st.text_input("Enter your query:")

if query:
    similar_docs = rag_system.get_similar_documents(query)
    if similar_docs:
        st.write("Similar documents found:")
        for doc in similar_docs:
            st.write(doc)
    else:
        st.write("No relevant documents found.")
