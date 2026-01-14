from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

raw_docs = [
    Document(
        page_content="RAG stands for Retrieval Augmented Generation. "
                     "It combines retrieval with generation.",
        metadata={"source": "rag_intro.txt"}
    ),
    Document(
        page_content="Embeddings convert text into numerical vectors "
                     "that capture semantic meaning.",
        metadata={"source": "embeddings.txt"}
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=5
)

def main():
    chunks = splitter.split_documents(raw_docs)

    for c in chunks:
        print(c.page_content, c.metadata)
