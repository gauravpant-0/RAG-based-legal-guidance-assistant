# RAG-Based Legal Guidance Assistant (India)

A Retrieval-Augmented Generation (RAG) based AI application designed to answer real-world legal questions using Bhartiya Nyaya Sanhita (BNS) provisions and Indian court case summaries.

This system retrieves relevant legal text from a curated knowledge base and generates grounded, context-aware responses, helping users understand legal concepts, procedures, and precedents without relying on LLM hallucinations.

# Project Objective

The objective of this project is to design and implement a domain-specific RAG application that:

Works on real legal documents

Uses vector-based retrieval for accuracy

Generates responses strictly grounded in source data

Demonstrates an end-to-end LLM + Vector DB pipeline

This project was built as part of a student-driven RAG application assignment, focusing on real-world applicability rather than toy examples.

# Domain

Legal & Compliance (India)

Specifically:

Bharatiya Nyaya Sanhita (BNS)

Categorized Indian court cases (multiple categories, multiple cases per category)

#How the System Works (RAG Flow)
Legal Text (.txt)
      ↓
Document Loader
      ↓
Text Chunking (Recursive Splitter)
      ↓
Embeddings (Sentence Transformers)
      ↓
Vector Store (FAISS)
      ↓
Similarity Retriever
      ↓
Prompt + Retrieved Context
      ↓
LLM (Groq – LLaMA 3.1)
      ↓
Grounded Legal Answer


The system ensures that:

Answers are generated only from retrieved legal text

Responses remain contextual and factual

Large legal documents are efficiently searchable

# Project Structure
RAG PROJECT/
│
├── data/
│   └── (optional future datasets)
│
├── outputs/
│   └── (vector store / cached outputs)
│
├── venv/
│   └── (virtual environment)
│
├── .env
│   └── GROQ_API_KEY
│
├── Bhartiya nyaya samhita.txt
│   └── Knowledge base (laws + court cases)
│
├── model.py
│   └── RAG pipeline (loading, embeddings, retrieval, LLM)
│
├── main.py
│   └── Streamlit user interface
│
└── README.md

# Knowledge Base

Single curated .txt file

Contains:

BNS legal provisions

Indian court case summaries

Clear category and case separators

Optimized for:

Semantic chunking

Accurate retrieval

Legal context preservation

# Tech Stack
Core Technologies

Python

LangChain

FAISS (Vector Database)

Groq API (LLaMA 3.1)

Sentence Transformers (Embeddings)

Streamlit (UI)

Models Used

Embedding Model: sentence-transformers/all-MiniLM-L6-v2

LLM: llama-3.1-8b-instant via Groq

# User Interface

Built using Streamlit

Allows users to:

Ask legal questions in natural language

Receive answers grounded in Indian legal text

Designed for:

Simplicity

Fast response

Educational and informational use

# Key Features

✔️ Domain-specific legal RAG system

✔️ Grounded responses (no free hallucination)

✔️ Optimized chunking for legal documents

✔️ Clean separation of backend and UI

✔️ Extendable to more laws or case data

⚠️ Disclaimer

This application:

Is for educational and informational purposes only

Does not provide legal advice

Should not be used as a substitute for a qualified legal professional

# Author

Gaurav Pant
CS Engineer | Context-AI Practitioner