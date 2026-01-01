from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-4o-mini",
    instructions=(
        "You are a senior Python engineer. "
        "Answer concisely and use bullet points when helpful."
    ),
    input="Explain what a virtual environment is.",
    temperature=0.0
    # temperature=0.9
)

print(resp.output_text)
