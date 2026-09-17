import os
from langchain_community.document_loaders import PyPDFLoader

PDF_FOLDER = "data/papers"

TITLE_MAP = {
    "lora.pdf": "LoRA: Low-Rank Adaptation of Large Language Models",
    "qlora.pdf": "QLoRA: Efficient Finetuning of Quantized LLMs",
    "rag.pdf": "Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks",
    "researchpaper 1.pdf": "Attention Is All You Need"
}

documents = []

for file in os.listdir(PDF_FOLDER):

    if file.endswith(".pdf"):

        path = os.path.join(PDF_FOLDER, file)

        loader = PyPDFLoader(path)

        pages = loader.load()

        for page in pages:

            page.metadata["paper_title"] = TITLE_MAP.get(
                file,
                os.path.splitext(file)[0]
            )

            page.metadata["page_number"] = (
                page.metadata.get("page", 0) + 1
            )

        documents.extend(pages)

print("Total pages loaded:", len(documents))

for doc in documents[:3]:

    print("\nPaper:", doc.metadata["paper_title"])
    print("Page:", doc.metadata["page_number"])
    print("Text:", doc.page_content[:200])