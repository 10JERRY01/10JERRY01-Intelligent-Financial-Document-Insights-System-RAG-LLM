# Intelligent Financial Document Insights System - Project Documentation

## Overview
The **Intelligent Financial Document Insights System** is an AI-powered tool designed to process, analyze, and query financial documents (e.g., annual reports). It combines exploratory data analysis (EDA), Retrieval-Augmented Generation (RAG), and a large language model (LLM) to extract trends and provide natural language answers to user queries. Built with Python, this project demonstrates skills in data science, machine learning, and visualization—tailored to financial applications.

## Objectives
- Extract and preprocess text from financial PDFs.
- Analyze document content to identify key trends (e.g., frequent terms).
- Enable intelligent querying using RAG and an LLM.
- Present insights through visualizations and an interactive interface.

## Features
1. **PDF Processing**: Extracts and cleans text from financial PDFs.
2. **Exploratory Data Analysis (EDA)**: Identifies top words and trends with statistical summaries.
3. **AI-Powered Querying**: Uses RAG and an LLM to answer questions based on document content.
4. **Interactive Dashboard**: Displays EDA visuals and query responses via Streamlit.


## Technical Details
### Dependencies
- Python 3.8+
- Libraries: `PyPDF2`, `pandas`, `nltk`, `matplotlib`, `seaborn`, `sentence-transformers`, `faiss-cpu`, `transformers`, `streamlit`, `torch`

### Workflow
1. **Data Pipeline** (`pdf_processor.py`):
   - Input: Financial PDF (e.g., JPMorgan Chase 2023 Annual Report).
   - Process: Extract text, clean noise, save as CSV.
2. **EDA** (`eda.py`):
   - Tokenize text, calculate word frequencies, visualize top 10 terms.
3. **RAG + LLM** (`rag_llm.py`):
   - Chunk text, generate embeddings with `all-MiniLM-L6-v2`, index with FAISS.
   - Query with RAG, generate answers using `distilgpt2`.
4. **Interface** (`app.py`):
   - Streamlit app combining EDA visuals and query functionality.

## Limitations
- **LLM Quality**: `distilgpt2` provides basic answers; larger models could improve accuracy.
- **Scalability**: Single-document focus; multi-document support requires extension.
- **Text Extraction**: Complex PDFs may yield imperfect text due to formatting.

## Future Enhancements
- Upgrade to a stronger LLM (e.g., GPT-3 via API).
- Add multi-document processing and comparison.
- Enhance UI with additional visualizations (e.g., word clouds).


## Author
[Rahul Das]  
(https://www.linkedin.com/in/rahul-das-9533a1195/)

# Intelligent Financial Document Insights System

An AI-powered tool to analyze and query financial documents, leveraging EDA, RAG, and LLMs. Built to showcase data science and machine learning skills for financial applications.

*Interactive dashboard displaying word frequency and query responses.*

## Features
- **PDF Processing**: Extracts and cleans text from financial PDFs.
- **EDA**: Visualizes key trends (e.g., top 10 words) with statistical insights.
- **AI Querying**: Answers questions using RAG and a lightweight LLM.
- **Interactive UI**: Streamlit app for easy exploration.

## Demo
- **Sample Query**: "What were the key revenue drivers in 2023?"
- **Output**: Retrieves relevant document sections and generates a natural language response.
- See the screenshots of the streamlit app.




