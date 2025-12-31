from openai import OpenAI

def main() -> None:
    # The SDK reads OPENAI_API_KEY automatically from your environment.
    client = OpenAI()

    resp = client.responses.create(
        model="gpt-4o-mini",
        input="Say hello in one short sentence and include one emoji.",
    )

    # resp.output_text is the easiest way to get the generated text.
    print(resp.output_text)

if __name__ == "__main__":
    main()
