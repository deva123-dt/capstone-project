from chunker import chunks
from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(model="nomic-embed-text")

vectorstore = Chroma(
    collection_name="research_papers",
    embedding_function=embeddings,
    persist_directory="vectorstore",
    collection_metadata={"hnsw:space": "cosine"}
)

batch_size = 25

for i in range(0, len(chunks), batch_size):
    batch = chunks[i:i + batch_size]
    vectorstore.add_documents(batch)
    print(f"Embedded {min(i + batch_size, len(chunks))}/{len(chunks)} chunks")

print("\nVECTOR DATABASE CREATED SUCCESSFULLY!")
print("Total chunks stored:", len(chunks))