from langchain_core.output_parsers import PydanticOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel

from src.agents.config import OPENAI_MODEL
from src.agents.prompts.agent import BASE_AGENT_SYSTEM_PROMPT, STRUCTURED_AGENT_SYSTEM_PROMPT

model = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0
)

prompt = ChatPromptTemplate.from_messages([
    ("system", BASE_AGENT_SYSTEM_PROMPT + "\n" + STRUCTURED_AGENT_SYSTEM_PROMPT),
    ("human", "User question: {user_input}\n\n"
              "{format_instructions}")
])

class AgentResponse(BaseModel):
    answer: str
    confidence: float
    action: str
    population: int

parser = PydanticOutputParser(pydantic_object=AgentResponse)

# prompt = prompt.partial(
#     format_instructions=parser.get_format_instructions()
# )

# model2 = ChatOpenAI(
#     model=OPENAI_MODEL,
#     temperature=0
# ).with_structured_output(AgentResponse)
#
# chain = prompt | model2

chain = prompt | model | parser

# OutputFixingParser
# RetryWithErrorOutputParser

def main():
    user_input = "What is the capital of France?"
    response = chain.invoke({
        "user_input": user_input,
        "format_instructions": parser.get_format_instructions(),
    })
    print(response)