from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from rank_bm25 import BM25Okapi

from chunker import chunks

# Dense retriever
embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma(
    collection_name="research_papers",
    embedding_function=embeddings,
    persist_directory="vectorstore"
)

query = "What is LoRA and how does it reduce trainable parameters?"

# Dense search
dense_results = vectorstore.similarity_search(
    query,
    k=5
)

# BM25 keyword search
corpus = [doc.page_content for doc in chunks]
tokenized_corpus = [text.lower().split() for text in corpus]

bm25 = BM25Okapi(tokenized_corpus)

tokenized_query = query.lower().split()

bm25_scores = bm25.get_scores(tokenized_query)

top_indices = sorted(
    range(len(bm25_scores)),
    key=lambda i: bm25_scores[i],
    reverse=True
)[:5]

bm25_results = [chunks[i] for i in top_indices]

# Combine results
combined = dense_results + bm25_results

# Remove duplicates
unique_results = []

seen = set()

for doc in combined:
    key = (
        doc.metadata.get("paper_title"),
        doc.metadata.get("page_number"),
        doc.page_content[:100]
    )

    if key not in seen:
        seen.add(key)
        unique_results.append(doc)

print("\nHYBRID SEARCH TOP RESULTS\n")

for i, doc in enumerate(unique_results[:3], 1):
    print(f"--- Result {i} ---")
    print("Paper:", doc.metadata.get("paper_title"))
    print("Page:", doc.metadata.get("page_number"))
    print("Content:", doc.page_content[:500])
    print()