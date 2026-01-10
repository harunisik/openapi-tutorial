from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnableMap, RunnableLambda

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

def get_context_for_question(question: str) -> str:
    # Return different context text depending on the question
    q = question.lower()
    if "rag" in q or "retrieval augmented generation" in q:
        return "RAG stands for Retrieval Augmented Generation."
    if "python" in q:
        return "Python is a high-level programming language used for many tasks."
    if "langchain" in q:
        return "LangChain is a framework for building LLM applications."
    return "No context available for this question."


chain = (
    RunnableMap({
        "question": lambda x: x["question"],
        "context": lambda x: get_context_for_question(x["question"]),
        "logger": lambda x: print(f"Received input: {x}")
    })
    | RunnableLambda(lambda x: (print(f"After RunnableMap: {x}"), x)[1])
    | prompt
    | RunnableLambda(lambda x: (print(f"Rendered prompt:\n{x}"), x)[1])
    | llm
    | RunnableLambda(lambda x: (print(f"LLM raw output: {x}"), x)[1])
    | StrOutputParser()
)


def main():
    print(chain.invoke({
        "question": "What does RAG stand for?",
    }))
