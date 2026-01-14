from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langchain.agents import create_agent

@tool
def get_definition(term: str) -> str:
    """Get a short technical definition."""
    return {
        "rag": "Retrieval Augmented Generation",
        "embedding": "A numeric vector representation of text",
    }.get(term.lower(), "Not found")

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

agent = create_agent(
    model=model,
    tools=[get_definition],
    system_prompt="Use tools when needed for accuracy."
)

def main():
    question = "What is RAG and embedding?"
    response = agent.invoke({"input": question})
    print(response)
