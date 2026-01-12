from typing import Any, Iterator

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain concepts clearly."),
    ("system", "{memory}"),
    ("user", "{question}")
])

memory = []

def load_history(_: Any) -> Iterator[str]:
    yield "\n".join(memory)

history_runnable = RunnableLambda(load_history)

inputs = RunnableMap({
    "memory": history_runnable,
    "question": RunnablePassthrough()
})

chain = inputs | prompt | llm | StrOutputParser()


def main():
    while True:
        user_question = input("Question (press Enter to quit): ").strip()
        if not user_question:
            print("Exiting...")
            break
        result = chain.invoke({"question": user_question})
        memory.append(f"User: {user_question}\nAI: {result}")
        print(result)
