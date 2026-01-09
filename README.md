# 🧠 RAG Resume Intelligence System

AI-powered resume screening and IT career reasoning platform using Retrieval-Augmented Generation.

![Banner](assets/banner7.png)

## Features
- Semantic resume-job matching (Sentence-BERT)
- FAISS vector knowledge base
- RAG chatbot using LLaMA-3.1 (Groq)
- Multi-format resume parsing
- Probability-based job fit scoring

## Architecture
Sentence-BERT → FAISS → LangChain RAG → Groq LLaMA → Streamlit UI

## Run Locally

```bash
pip install -r requirements.txt
streamlit run app.py
