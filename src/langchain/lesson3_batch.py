from langchain_openai import ChatOpenAI

questions = [
    "What is RAG?",
    "What is an embedding?",
    "What is a vector database?"
]

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

def main():
    responses = llm.batch(questions)
    for r in responses:
        print(r.content)
