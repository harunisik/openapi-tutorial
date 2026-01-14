from langchain_core.vectorstores import InMemoryVectorStore
from langchain_openai import OpenAIEmbeddings

embeddings = OpenAIEmbeddings(
    model="text-embedding-3-small"
)


texts = [
    "RAG stands for Retrieval Augmented Generation.",
    "Embeddings represent text numerically.",
    "Vector databases enable semantic search."
]

vectorstore = InMemoryVectorStore.from_texts(
    texts=texts,
    embedding=embeddings
)

def main():
    vector = embeddings.embed_query("What is RAG?")
    print(len(vector))   # e.g. 1536

    docs = vectorstore.similarity_search(
        "What does RAG mean?",
        k=2
    )

    for d in docs:
        print(d.page_content)

