from __future__ import annotations
from openai import OpenAI

def main() -> None:
    # The SDK reads OPENAI_API_KEY automatically from your environment.
    client = OpenAI()

    resp = client.responses.create(
        model="gpt-4o-mini",
        # input="Say hello in one short sentence and include one emoji.",
        input="Give me 3 alternative greetings, each < 6 words.",
    )

    # resp.output_text is the easiest way to get the generated text.
    # 1) Get the generated text (best default)
    print("OUTPUT_TEXT:\n", resp.output_text)

    # 2) Token usage (cost + limits awareness)
    print("\nUSAGE:", resp.usage)

    # 3) Debug/trace: request id + response id
    # (request id is useful when contacting support)
    print("\nRESPONSE_ID:", resp.id)
    print("REQUEST_ID:", getattr(resp, "_request_id", None))

    # 4) If you *need* JSON for logging:
    # model_dump_json is the safe way to serialize SDK objects
    print("\nJSON:\n", resp.model_dump_json(indent=2))

if __name__ == "__main__":
    main()
