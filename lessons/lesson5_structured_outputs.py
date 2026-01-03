import json
from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-4o-mini",
    instructions=(
        "You are a data extraction API.\n"
        "Rules:\n"
        "- Output valid JSON only\n"
        "- Do not include explanations\n"
        "- Follow the schema exactly"
    ),
    input=(
        "Extract user info:\n"
        "Name: Alice\n"
        "Age: 30"
    ),
    temperature=0.0,
    max_output_tokens=100,
)

data = json.loads(resp.output_text)
print(data)
