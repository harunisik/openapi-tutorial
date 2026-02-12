from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

from src.langchain.rag_demo.ingest import EMBEDDING_MODEL_NAME

raw_docs = [
    Document(
        page_content=
                 "CRISPR is a gene-editing technology."
                 "CRISPR allows targeted modifications of DNA."
                 "CRISPR originated from bacterial immune systems.",
        metadata={"source": "crispr.txt"}
    )
]

splitter = RecursiveCharacterTextSplitter(
    chunk_size=25,
    chunk_overlap=5
)

chunks = splitter.split_documents(raw_docs)

embeddings = OpenAIEmbeddings(
    model=EMBEDDING_MODEL_NAME
)

vectorstore = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)