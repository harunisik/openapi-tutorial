from openai import OpenAI

client = OpenAI()

resp = client.responses.create(
    model="gpt-4o-mini",
    input="There are 8 tokens in this sentence.",
    max_output_tokens=16,
    temperature=0.0,
)

print(resp.output_text)
print("Usage:", resp.usage)
print("Input tokens used:", resp.usage.input_tokens)
print("Output tokens used:", resp.usage.output_tokens)
print("Total tokens used:", resp.usage.total_tokens)
