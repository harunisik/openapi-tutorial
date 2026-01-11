from langchain_core.output_parsers import StructuredOutputParser
from langchain_core.output_parsers import ResponseSchema
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI

llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0
)

# {
#     "answer": "Retrieval Augmented Generation",
#     "confidence": 0.95
# }
schemas = [
    ResponseSchema(
        name="answer",
        description="The answer to the question"
    ),
    ResponseSchema(
        name="confidence",
        description="Confidence from 0 to 1"
    ),
]

parser = StructuredOutputParser.from_response_schemas(schemas)
format_instructions = parser.get_format_instructions()

prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question."),
    ("user", "{question}\n\n{format_instructions}")
])

chain = (prompt
         | llm
         | parser
         )

def main():
    result = chain.invoke({
        "question": "What does RAG stand for?",
        "format_instructions": format_instructions,
    })

    print(result)
