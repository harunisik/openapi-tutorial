from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "Answer using only the provided context. "
        "If unsure, say 'I don't know'."
    ),
    ("user", "Context:\n{context}\n\n"
             "Question:\n{question}")
])

chain = (
    RunnableMap({
        "question": lambda x: x["question"],
        "context": lambda x: x["context"],
    })
    | prompt
    | llm
    | StrOutputParser()
)

def main():
    print(chain.invoke({
        "question": "What does RAG stand for?",
        "context": "RAG stands for Retrieval Augmented Generation."
    }))
