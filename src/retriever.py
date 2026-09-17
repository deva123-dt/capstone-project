from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma(
    collection_name="research_papers",
    embedding_function=embeddings,
    persist_directory="vectorstore"
)

query = "What is LoRA and how does it reduce trainable parameters?"

results = vectorstore.similarity_search_with_score(
    query,
    k=3
)

print("\nTOP 3 RESULTS\n")

for i, (doc, score) in enumerate(results, 1):
    print(f"--- Result {i} ---")
    print("Paper:", doc.metadata.get("paper_title"))
    print("Page:", doc.metadata.get("page_number"))
    print("Score:", score)
    print("Content:", doc.page_content[:500])
    print()