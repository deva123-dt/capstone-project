from langchain_ollama import OllamaEmbeddings

query = "What is LoRA and how does it reduce trainable parameters?"

models = [
    "nomic-embed-text",
    "mxbai-embed-large"
]

print("\nEMBEDDING MODEL COMPARISON\n")

for model in models:
    embeddings = OllamaEmbeddings(model=model)

    vector = embeddings.embed_query(query)

    print("Model:", model)
    print("Dimensions:", len(vector))
    print("First 5 values:", vector[:5])
    print()