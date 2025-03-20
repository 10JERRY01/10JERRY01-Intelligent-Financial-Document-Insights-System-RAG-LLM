# Install libraries if needed: pip install transformers sentence-transformers faiss-cpu torch
import pandas as pd
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
from transformers import pipeline


# Step 1: Load and chunk the text
def load_and_chunk_text(csv_path, chunk_size=500):
    df = pd.read_csv(csv_path)
    text = df['text'][0]
    # Split into chunks of roughly chunk_size characters
    chunks = [text[i:i + chunk_size] for i in range(0, len(text), chunk_size)]
    return chunks


# Step 2: Generate embeddings and build FAISS index
def build_faiss_index(chunks):
    # Load a pre-trained sentence transformer model
    encoder = SentenceTransformer('all-MiniLM-L6-v2')  # Lightweight and effective
    embeddings = encoder.encode(chunks, convert_to_numpy=True)

    # Create FAISS index
    dimension = embeddings.shape[1]  # Embedding size
    index = faiss.IndexFlatL2(dimension)  # L2 distance index
    index.add(embeddings)

    return index, embeddings, encoder


# Step 3: Query the document with RAG
def query_document(query, chunks, index, embeddings, encoder, top_k=3):
    # Encode the query
    query_embedding = encoder.encode([query], convert_to_numpy=True)

    # Search for top_k most similar chunks
    distances, indices = index.search(query_embedding, top_k)
    retrieved_chunks = [chunks[idx] for idx in indices[0]]

    # Combine retrieved chunks into context
    context = " ".join(retrieved_chunks)
    return context


# Step 4: Use an LLM to generate an answer
def generate_answer(query, context):
    # Load a lightweight LLM for text generation
    llm = pipeline('text-generation', model='distilgpt2', max_new_tokens=50)
    prompt = f"Question: {query}\nContext: {context}\nAnswer:"
    response = llm(prompt)[0]['generated_text']
    # Extract just the answer part (post-prompt)
    answer = response.split("Answer:")[-1].strip()
    return answer


# Example usage
csv_path = "processed_text.csv"
chunks = load_and_chunk_text(csv_path)
index, embeddings, encoder = build_faiss_index(chunks)

# Test a query
query = "What were the key revenue drivers in 2023?"
context = query_document(query, chunks, index, embeddings, encoder)
answer = generate_answer(query, context)

print(f"Query: {query}")
print(f"Retrieved Context: {context[:500]}...")  # Truncated for brevity
print(f"Answer: {answer}")