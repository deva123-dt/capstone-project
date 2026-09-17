from loader import documents
from langchain_text_splitters import RecursiveCharacterTextSplitter

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=50
)

chunks = splitter.split_documents(documents)

print("Total pages:", len(documents))
print("Total chunks:", len(chunks))

for chunk in chunks[:3]:
    print("\nPaper:", chunk.metadata["paper_title"])
    print("Page:", chunk.metadata["page_number"])
    print("Chunk:", chunk.page_content[:300])