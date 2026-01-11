from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field
from langchain_core.output_parsers import PydanticOutputParser

document = [
    {"id": "doc1", "text": "RAG stands for Retrieval-Augmented Generation."},
    {"id": "doc2", "text": "It combines pre-trained language models with external knowledge sources."},
    {"id": "doc3", "text": "This approach improves the accuracy and relevance of generated responses."},
]

class RAGAnswer(BaseModel):
        answer: str = Field(description="Grounded answer")
        citations: list[str] = Field(description="Source IDs")
        # sources: list[str] = Field(description="At least one source ID", min_length=1)
        confidence: float = Field(ge=0, le=1)
        missing_info: bool = Field(description="True if information is missing to answer the question fully")

parser = PydanticOutputParser(pydantic_object=RAGAnswer)

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using only the provided context. "
               "If the information is not available, say 'I don't know'. "),
    ("user", "{context}\n\n"
             "{question}\n\n"
             "{format_instructions}")
])

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# Key line: use OpenAI native structured outputs
# structured_llm = llm.with_structured_output(RAGAnswer, method="json_schema")

chain = prompt | llm | parser

def main():
    result = chain.invoke({
        "context": "\n".join([f"[{doc['id']}] {doc['text']}" for doc in document]),
        "question": "Who invented RAG and what problem does it solve?",
        "format_instructions": parser.get_format_instructions(),
    })
    print(result)
