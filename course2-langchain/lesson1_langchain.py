from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
  ("system", "You are a precise technical assistant."),
  ("user", "{question}")
])

model = ChatOpenAI(
  model="gpt-4.1-mini",
  temperature=0,
)

chain = prompt | model

result = chain.invoke({"question": "Explain LangChain in 2 sentences."})
print(result.content)
