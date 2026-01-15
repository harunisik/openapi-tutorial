import os

from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.vectorstores import InMemoryVectorStore

load_dotenv()

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE"))
OVERLAP_SIZE = int(os.getenv("OVERLAP_SIZE"))
EMBEDDING_MODEL_NAME = os.getenv("EMBEDDING_MODEL_NAME")

def ingest_documents(raw_documents) -> InMemoryVectorStore:
    """
    Ingest a list of raw documents into the vector store.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=OVERLAP_SIZE
    )

    chunks = splitter.split_documents(raw_documents)

    embeddings = OpenAIEmbeddings(
        model=EMBEDDING_MODEL_NAME
    )

    vectorstore = InMemoryVectorStore.from_documents(
        documents=[],
        embedding=embeddings
    )

    vectorstore.add_documents(chunks)

    return vectorstore