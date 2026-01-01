from openai import OpenAI

client = OpenAI()

PROMPT_INSTRUCTIONS = """
You are a precise API assistant.

Rules:
- Respond with valid JSON only
- Do not include explanations
- Follow the schema exactly
"""

def run(prompt: str) -> str:
    resp = client.responses.create(
        model="gpt-4o-mini",
        instructions=PROMPT_INSTRUCTIONS,
        input=prompt,
        temperature=0.0,
        max_output_tokens=200,
    )
    return resp.output_text

if __name__ == "__main__":
    prompt = (
        "Generate a JSON object with the following schema:\n"
        "{\n"
        '  "greeting": string, // A one-sentence greeting\n'
        '  "language": string  // The language of the greeting like tr, fr, es\n'
        "}\n"
    )
    output = run(prompt)
    print(output)