from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI

Document(
    page_content="RAG stands for Retrieval Augmented Generation...",
    metadata={
        "doc_id": "rag_intro",
        "source": "rag_intro.txt",
        "chunk": 1
    }
)

splitter = RecursiveCharacterTextSplitter(
    chunk_size=500,
    chunk_overlap=100
)

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)

vectorstore = InMemoryVectorStore.from_documents(
    documents=[],  # start empty
    embedding=embeddings
)

def ingest_documents(raw_documents):
    chunks = splitter.split_documents(raw_documents)
    vectorstore.add_documents(chunks)

# str → list[Document]
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 4}
)

def format_docs(docs):
    lines = []
    for d in docs:
        src = d.metadata.get("source", "unknown")
        lines.append(f"[{src}] {d.page_content}")
    return "\n\n".join(lines)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a factual assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not present, say 'I don't know'."
    ),
    (
        "user",
        "Context:\n{context}\n\nQuestion:\n{question}"
    )
])

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

inputs = RunnableMap({
    "question": RunnablePassthrough(),
    "context": retriever | format_docs,
})

rag_chain = chain = inputs | prompt | llm | StrOutputParser()

def main():
    answer = rag_chain.invoke("What does RAG stand for?")
    print(answer)
