from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

def main():
    response = llm.invoke(
        [
            SystemMessage(content="You explain concepts clearly."),
            HumanMessage(content="What is RAG in one sentence?")
        ]
    )
    print(response.model_dump_json())