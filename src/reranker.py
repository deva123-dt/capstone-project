from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

from chunker import chunks
import numpy as np

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma(
    collection_name="research_papers",
    embedding_function=embeddings,
    persist_directory="vectorstore"
)

query = "What is LoRA and how does it reduce trainable parameters?"

results = vectorstore.similarity_search(query, k=10)

query_vector = np.array(
    embeddings.embed_query(query)
)

scored_results = []

for doc in results:
    doc_vector = np.array(
        embeddings.embed_query(doc.page_content)
    )

    similarity = np.dot(query_vector, doc_vector) / (
        np.linalg.norm(query_vector) *
        np.linalg.norm(doc_vector)
    )

    scored_results.append((doc, similarity))

scored_results.sort(
    key=lambda x: x[1],
    reverse=True
)

print("\nRERANKED TOP 3 RESULTS\n")

for i, (doc, score) in enumerate(scored_results[:3], 1):
    print(f"--- Result {i} ---")
    print("Paper:", doc.metadata.get("paper_title"))
    print("Page:", doc.metadata.get("page_number"))
    print("Rerank Score:", round(float(score), 4))
    print("Content:", doc.page_content[:500])
    print()