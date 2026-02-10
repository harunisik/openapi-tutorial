import os

from dotenv import load_dotenv
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableMap, RunnablePassthrough
from langchain_openai import ChatOpenAI

from document import format_docs
from document import read_document
from ingest import ingest_documents

load_dotenv()

OPENAI_MODEL = os.getenv("OPENAI_MODEL")

document = read_document("data/Identifier+Descriptions.txt")
vectorstore = ingest_documents([document])
retriever = vectorstore.as_retriever(
    search_kwargs={"k": 5}
)
prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer using ONLY the provided context. "
               "If the answer is not in the context, say 'I don't know'."),
    ("user", "Context:\n{context}\n\nQuestion:\n{question}")
])
inputs = RunnableMap({
    "question": RunnablePassthrough(),
    "context": (retriever
                # | RunnableLambda(lambda docs: (print(json.dumps([{"page_content": d.page_content, "metadata": d.metadata} for d in docs], indent=2, default=str)), docs)[1])
                | format_docs),
})
llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0,
    max_tokens=100
)

chain = (inputs
         | prompt
         | llm
         | StrOutputParser())

def main():
    while True:
        user_question = input("Question (press Enter to quit): ").strip()
        if not user_question:
            print("Exiting...")
            break
        result = chain.invoke(user_question)
        print(result)

if __name__ == "__main__":
  print("----------- RAG Demo Module -----------\n\n")
  main()

# messages=[SystemMessage(content="Answer using ONLY the provided context. If the answer is not in the context, say 'I don't know'.", additional_kwargs={}, response_metadata={}),
# HumanMessage(content='Context:\nISIN\nThe International Securities Identification Number (ISIN) is a unique 12-character alphanumeric code used to identify securities, such as stocks, bonds, and other financial instruments. The ISIN code ensures a uniform and globally accepted standard for identifying securities, which facilitates efficient trading, clearing, and settlement processes across different markets and systems.\nStructure of an ISIN\n\nStructure of an ISIN\n1. Country Code (2 characters): The first two characters represent the country where the issuing entity is headquartered, according to ISO 3166-1 alpha-2 country codes. For example, "US" for the United States, "GB" for the United Kingdom.\n\nIdentifier Descriptions\n* General description\n* ISIN \no Structure of an ISIN\n* Bloomberg Ticker \no Structure of an BB Ticker\no Security Identifier List:\nGeneral description\nThis page is dedicated to common identifiers used within the structured products industrie.\nISIN\n\n3. Check Digit (1 character): The final character is a check digit, which is used to verify the accuracy of the ISIN. It is calculated using the Luhn algorithm, ensuring the ISIN is correctly formed and valid.\nBloomberg Ticker\n\n3. Security Identifier: Additional characters or codes may be added to indicate specific types of securities, such as common stock, preferred stock, or bonds. For instance, a common stock might have a ticker like "AAPL US Equity," while a bond might have a different identifier attached to the base symbol.\nSecurity Identifier List:\nmost common for us:\n* EQUITY - common�shares\n* COMDTY - commodity markets�\n* INDEX - indices\n* CRNCY - currency markets\nless common:\n\nQuestion:\nWhat is ISIN?', additional_kwargs={}, response_metadata={})]

# Optional exercises (strongly recommended)
#
# Add doc_id + citations to output
# Add a query-rewrite step before retrieval
# Add memory summary to the RAG chain
# Swap FAISS → Pinecone (code should barely change)

# Optional exercises
#
# Implement query rewrite only when question length > N chars
# Switch retriever to MMR and compare results
# Add a simple in-memory cache for retrieval results
# Measure latency breakdown (retrieve vs LLM)

# Optional exercises
#
# Add a guard that rejects empty context
# Log prompt + retrieved docs for one query
# Enable LangSmith and inspect a trace
# Simulate a broken document update and observe behavior