from langchain_core.runnables import RunnableLambda
from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langsmith.testing import log_inputs

prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain concepts clearly."),
    ("user", "{question}")
])

model = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

log_inputs = RunnableLambda(
    lambda inputs: (
        print("--------------------------\n"
              f"Input before model: {inputs}\n"
              "-------------------------\n") or inputs
    )
)

return_raw_message = RunnableLambda(lambda msg: msg)

chain = prompt | log_inputs | model | return_raw_message

def main():
    result = chain.invoke({"question": "What is RAG in one sentence?"})
    print(result)
