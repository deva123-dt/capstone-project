from langchain_ollama import OllamaEmbeddings, ChatOllama
from langchain_chroma import Chroma

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)

vectorstore = Chroma(
    collection_name="research_papers",
    embedding_function=embeddings,
    persist_directory="vectorstore"
)

llm = ChatOllama(
    model="gemma3:1b",
    temperature=0
)

query = "What is LoRA and how does it reduce trainable parameters?"

results = vectorstore.similarity_search(query, k=3)

context = "\n\n".join(
    [
        f"Paper: {doc.metadata.get('paper_title')}\n"
        f"Page: {doc.metadata.get('page_number')}\n"
        f"Content: {doc.page_content}"
        for doc in results
    ]
)

prompt = f"""
Answer the question using ONLY the research paper context below.

Question:
{query}

Research Paper Context:
{context}

Give a clear and concise answer.
"""

response = llm.invoke(prompt)

print("\nANSWER:\n")
print(response.content)

print("\nTOP 3 SOURCES:\n")

for i, doc in enumerate(results, 1):
    print(
        f"{i}. {doc.metadata.get('paper_title')} "
        f"- Page {doc.metadata.get('page_number')}"
    )