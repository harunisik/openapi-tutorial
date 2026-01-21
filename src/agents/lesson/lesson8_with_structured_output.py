from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
from pydantic import BaseModel, Field
from typing import Literal, Optional, Dict, Any

class CreateCustomer(BaseModel):
    action: Literal["create_customer"] = "create_customer"
    email: str
    name: Optional[str] = None
    phone: Optional[str] = None

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Extract the customer fields for our API call."),
    ("human", "User message: {text}") # No format instructions needed with Pydantic model
])

structured_llm = llm.with_structured_output(CreateCustomer) # Using Pydantic model for structured output

chain = prompt | structured_llm

def main():
    obj = chain.invoke({"text": "Create a customer for harun@example.com, name Harun Isik, tag source=mobile"})
    print(obj)  # Pydantic instance
