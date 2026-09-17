import csv
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


results_data = []


with open(
    "experiments/test_questions.csv",
    "r",
    encoding="utf-8"
) as file:

    questions = csv.DictReader(file)

    for number, row in enumerate(questions, 1):

        query = row["question"]

        results = vectorstore.similarity_search(
            query,
            k=3
        )

        context = "\n\n".join(
            f"Paper: {doc.metadata.get('paper_title')}\n"
            f"Page: {doc.metadata.get('page_number')}\n"
            f"Content: {doc.page_content}"
            for doc in results
        )

        prompt = f"""
Answer the question using ONLY the research paper context.

Question:
{query}

Context:
{context}

If the answer is not present in the context, say:
"I could not find the answer in the provided research papers."

Give a clear and concise answer.
"""

        response = llm.invoke(prompt)

        sources = " | ".join(
            f"{doc.metadata.get('paper_title')} - Page {doc.metadata.get('page_number')}"
            for doc in results
        )

        results_data.append({
            "question": query,
            "answer": response.content,
            "top_3_sources": sources
        })

        print(f"\nQUESTION {number}: {query}")
        print("ANSWER:", response.content)
        print("SOURCES:", sources)


with open(
    "experiments/evaluation_results.csv",
    "w",
    newline="",
    encoding="utf-8"
) as file:

    writer = csv.DictWriter(
        file,
        fieldnames=[
            "question",
            "answer",
            "top_3_sources"
        ]
    )

    writer.writeheader()
    writer.writerows(results_data)


print("\n================================")
print("EVALUATION COMPLETED")
print("Results saved to:")
print("experiments/evaluation_results.csv")
print("================================")