from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    streaming=True
)

def main():
    for chunk in llm.stream("Explain embeddings simply."):
        print(chunk.content, end="", flush=True)