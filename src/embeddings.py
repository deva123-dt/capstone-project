from langchain_ollama import OllamaEmbeddings

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vector = embeddings.embed_query(
    "What is LoRA in large language models?"
)

print("EMBEDDING CREATED SUCCESSFULLY!")
print("Embedding dimensions:", len(vector))
print("First 10 values:", vector[:10])