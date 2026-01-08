from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain concepts clearly."),
    ("user", "{question}")
])

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

parser = StrOutputParser()

chain = prompt | model | parser

result = chain.invoke({"question": "What is RAG in one sentence?"})
print(result)
