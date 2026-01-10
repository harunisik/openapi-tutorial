from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

document = [
    {"id": "doc1", "text": "RAG stands for Retrieval-Augmented Generation."},
    {"id": "doc2", "text": "It combines pre-trained language models with external knowledge sources."},
    {"id": "doc3", "text": "This approach improves the accuracy and relevance of generated responses."},
]

base_prompt = ChatPromptTemplate.from_messages([
    ("system", "{system_rules}"),
    ("user", "Context:\n{context}\n\n"
             "Question:\n{question}")
])

rag_prompt_strict = base_prompt.partial(
    system_rules=(
        "You are a factual assistant. "
        "Answer ONLY using the provided context. "
        "If the answer is not in the context, say 'I don't know'. "
        "Cite answers using the document IDs, e.g., [doc1]."
    )
)

rag_prompt_relax = base_prompt.partial(
    system_rules=(
        "You are a factual assistant. "
        "Answer using the provided context when possible. "
        "If the answer is not in the context, you may use your own knowledge. "
    )
)

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

chain = rag_prompt_relax | llm | StrOutputParser()


def main() -> None:
    result = chain.invoke({
        "context": "\n".join([f"[{doc['id']}] {doc['text']}" for doc in document]),
        "question": "What does RAG do?",
    })
    print(result)
