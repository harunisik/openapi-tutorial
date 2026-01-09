import langchain_core.prompts as ChatPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question in 3 bullet points."),
    ("user", "{question}"),
])

llm = ChatOpenAI(
  model = "gpt-4.1-mini",
  temperature = 0,
)

chain = prompt | llm | StrOutputParser()

def main() -> None:
  result = chain.invoke({"question": "What is RAG?"})
  print(result)
