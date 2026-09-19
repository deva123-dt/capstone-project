# 📚 Research Paper Answer Bot

An AI-powered Research Paper Question Answering system built using Retrieval-Augmented Generation (RAG).

## 🎯 Objective

The system answers questions from selected Generative AI research papers using semantic retrieval, hybrid search, reranking, and a local Large Language Model.

The system retrieves relevant research-paper content from a vector database and uses the retrieved context to generate grounded answers.

## 📄 Research Papers

The dataset contains four seminal Generative AI research papers:

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
- mxbai-embed-large
- Gemma 3 1B
- BM25
- Cosine Similarity
- Streamlit
- Retrieval-Augmented Generation (RAG)

## 📊 Dataset Statistics

- Research Papers: 4
- Total Pages: 86
- Total Chunks: 649
- Embedding Models Tested: 2
- Test Questions: 10
- Vector Database: ChromaDB
- LLM: Gemma 3 1B

## 🔄 RAG Pipeline

```text
PDF Research Papers
        ↓
Document Loading
        ↓
Text Chunking
        ↓
Embedding Generation
        ↓
ChromaDB Vector Database
        ↓
Dense / MMR Retrieval
        ↓
BM25 Keyword Retrieval
        ↓
Hybrid Retrieval
        ↓
Cosine Similarity Reranking
        ↓
Context Creation
        ↓
Gemma 3 1B
        ↓
Grounded Answer
        ↓
Top-3 Sources + Page Numbers