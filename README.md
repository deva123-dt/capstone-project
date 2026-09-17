# 📚 Research Paper Answer Bot

An AI-powered Research Paper Question Answering system built using Retrieval-Augmented Generation (RAG).

## 🎯 Objective

The system answers questions from selected Generative AI research papers using semantic retrieval, hybrid search, reranking, and a local Large Language Model.

## 📄 Research Papers

1. LoRA: Low-Rank Adaptation of Large Language Models
2. QLoRA: Efficient Finetuning of Quantized LLMs
3. Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks
4. Attention Is All You Need

## 🛠️ Technologies

- Python
- LangChain
- ChromaDB
- Ollama
- Nomic Embed Text
- Gemma 3 1B
- BM25
- Cosine Similarity
- Streamlit
- RAG

## 🔄 RAG Pipeline

PDF Research Papers
→ Document Loading
→ Text Chunking
→ Embedding Generation
→ ChromaDB
→ Dense / MMR / Hybrid Retrieval
→ Cosine Similarity Reranking
→ Context Creation
→ Gemma 3 1B
→ Grounded Answer
→ Top-3 Sources

## 🔎 Retrieval Methods

### Dense Retrieval
Uses semantic similarity between query and document embeddings.

### MMR Retrieval
Retrieves relevant and diverse document chunks.

### Hybrid Search
Combines Dense Retrieval with BM25 keyword search.

### Cosine Similarity Reranking
Reranks retrieved documents based on embedding similarity.

## 📊 Project Statistics

- Research Papers: 4
- Total Pages: 86
- Total Chunks: 649
- Embedding Models Tested: 2
- Test Questions: 10
- LLM: Gemma 3 1B
- Vector Database: ChromaDB

## ✨ Features

- Research paper question answering
- Top-3 source attribution
- Page number citation
- Multiple retrieval methods
- Conversational memory
- Failure case handling
- Streamlit web interface

## 🚀 How to Run

Install dependencies:

```bash
pip install -r requirements.txt
## 🔬 Embedding Experiment

Two embedding models were experimentally evaluated in the local environment:

- Nomic Embed Text
- mxbai-embed-large

Both models were executed locally through Ollama.

### Limitation

A commercial embedding API was not evaluated because an external commercial API key was not available in the development environment. Therefore, the project reports only the embedding models that were actually executed and tested.