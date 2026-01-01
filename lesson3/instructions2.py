from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-4o-mini",
    instructions=(
        "You are a precise assistant. "
        "Do not add extra commentary."
    ),
    input="List 3 benefits of Python.",
    temperature=0.0,
    max_output_tokens=100,
)

print(resp.output_text)
print("Usage:", resp.usage)