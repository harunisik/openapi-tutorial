from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a precise technical assistant."),
    ("user", "Explain {topic} in 2 sentences for {audience}.")
])

model = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
)

chain = prompt | model

topic = "LangChain"
audience = "a busy developer"

# Format messages and log the exact messages that will be sent to the model
messages = prompt.format_messages(topic=topic, audience=audience)

print("Messages sent to model:")
for msg in messages:
    role = getattr(msg, "role", getattr(msg, "type", None)) or "<unknown role>"
    content = getattr(msg, "content", None)
    # Fallback to string representation if content is not available
    if content is None:
        content = str(msg)
    print(f"{role}: {content}")

def main():
    # Invoke the chain with the two inputs
    result = chain.invoke({"topic": topic, "audience": audience})
    print(result.content)