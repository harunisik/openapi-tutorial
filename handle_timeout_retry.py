from __future__ import annotations

from openai import OpenAI

def make_client() -> OpenAI:
  # The SDK automatically reads OPENAI_API_KEY from env.
  # We keep initialization in one place for consistency.
  return OpenAI(
    # You can also pass api_key=os.environ["OPENAI_API_KEY"]
    # but env var is cleaner.
    timeout=30.0,   # seconds (avoid hanging forever)
    max_retries=2,  # simple built-in retry for transient issues
  )

client = make_client()

resp = client.responses.create(
  model="gpt-4o-mini",
  input="Return a one-sentence greeting.",
)

print(resp.output_text)
