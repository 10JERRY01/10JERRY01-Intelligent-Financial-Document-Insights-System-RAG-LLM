import streamlit as st
import pandas as pd
import nltk
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns


# Delay heavy imports until needed
def load_and_chunk_text(csv_path, chunk_size=500):
    df = pd.read_csv(csv_path)
    text = df['text'][0]
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


def build_faiss_index(chunks):
    from sentence_transformers import SentenceTransformer
    import faiss
    import numpy as np
    encoder = SentenceTransformer('all-MiniLM-L6-v2')
    embeddings = encoder.encode(chunks, convert_to_numpy=True)
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index, embeddings, encoder


def query_document(query, chunks, index, embeddings, encoder, top_k=3):
    import numpy as np
    query_embedding = encoder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]
    return " ".join(retrieved_chunks)


def generate_answer(query, context):
    from transformers import pipeline
    llm = pipeline('text-generation', model='distilgpt2', max_new_tokens=50)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    response = llm(prompt)[0]['generated_text']
    return response.split("Answer:")[-1].strip()


def analyze_text(text):
    words = word_tokenize(text.lower())
    stop_words = set(stopwords.words('english'))
    words = [word for word in words if word.isalpha() and word not in stop_words]
    word_freq = Counter(words).most_common(10)
    return word_freq


# Streamlit app
def main():
    st.title("Intelligent Financial Document Insights System")
    st.write("Analyze and query financial documents with AI!")

    # Load data
    csv_path = "processed_text.csv"
    text = pd.read_csv(csv_path)['text'][0]
    chunks = load_and_chunk_text(csv_path)
    index, embeddings, encoder = build_faiss_index(chunks)

    # Section 1: EDA Visualization
    st.header("Exploratory Data Analysis")
    word_freq = analyze_text(text)
    words, counts = zip(*word_freq)

    fig, ax = plt.subplots(figsize=(10, 6))
    sns.barplot(x=list(counts), y=list(words), ax=ax)
    ax.set_title('Top 10 Most Frequent Words')
    ax.set_xlabel('Frequency')
    ax.set_ylabel('Words')
    st.pyplot(fig)

    # Section 2: Query Interface
    st.header("Ask a Question")
    query = st.text_input("Enter your question (e.g., 'What were the key revenue drivers?')")
    if st.button("Get Answer"):
        if query:
            context = query_document(query, chunks, index, embeddings, encoder)
            answer = generate_answer(query, context)
            st.write("**Retrieved Context (Snippet):**", context[:500] + "...")
            st.write("**Answer:**", answer)
        else:
            st.write("Please enter a question!")


if __name__ == "__main__":
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    main()