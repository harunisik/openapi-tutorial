from langchain_core.messages import ToolMessage
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


@tool
def get_definition(term: str) -> str:
    """Get a short definition for a technical term."""
    definitions = {
        "rag": "Retrieval Augmented Generation",
        "embedding": "A numeric representation of text",
    }
    return definitions.get(term.lower(), "Definition not found")

@tool
def get_current_time() -> str:
    """Get the current time as a string."""
    from datetime import datetime
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S test")

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)
llm_with_tools = llm.bind_tools([get_current_time])

prompt = ChatPromptTemplate.from_messages([
    ("system", "Use tools when they help answer accurately."),
    ("user", "{question}")
])

chain = prompt | llm_with_tools | StrOutputParser()

def main():
    response = llm_with_tools.invoke("What time is it?")

    if response.tool_calls:
        print("Invoking tool...")

        tool_call = response.tool_calls[0]
        tool_result = get_current_time.invoke(tool_call["args"])
        print(f"Tool result: {tool_result}")

        tool_message = ToolMessage(
            content=tool_result,
            tool_call_id=tool_call["id"]
        )

        final_response = llm_with_tools.invoke([
            response,
            tool_message
        ])

        print(final_response.content)
    else:
        print("No tool calls made.")
        print(response.content)
