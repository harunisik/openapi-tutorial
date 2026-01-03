from openai import OpenAI

client = OpenAI()

# 3) Few-shot prompting (show, don’t tell)
resp = client.responses.create(
    model="gpt-4o-mini",
    instructions=(
        "Classify sentiment as POSITIVE or NEGATIVE.\n\n"
        "Example:\n"
        "Input: I love this product\n"
        "Output: POSITIVE\n\n"
        "Example:\n"
        "Input: This is terrible\n"
        "Output: NEGATIVE"
    ),
    input="Input: The experience was disappointing",
    temperature=0.0,
    max_output_tokens=20,
)

print(resp.output_text)
