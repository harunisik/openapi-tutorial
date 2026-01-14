from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough, RunnableLambda
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
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

chunks = splitter.split_documents(raw_docs)

for c in chunks:
    print(c.page_content, c.metadata)
print("---")

embeddings = OpenAIEmbeddings(model="text-embedding-3-small")

vectorstore = InMemoryVectorStore.from_documents(
    documents=chunks,
    embedding=embeddings
)

retriever = vectorstore.as_retriever(
    search_kwargs={"k": 3}
)

def format_docs(docs):
    return "\n\n".join(d.page_content for d in docs)

inputs = RunnableMap({
    "question": RunnablePassthrough(),
    "context": retriever | format_docs,
})

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the provided context. "
               "If the answer is not in the context, say 'I don't know'."),
    ("user", "Context:\n{context}\n\nQuestion:\n{question}")
])

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

def main():
    chain = inputs | prompt | llm | StrOutputParser()
    print(chain.invoke("What does RAG stand for?"))