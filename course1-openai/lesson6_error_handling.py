from __future__ import annotations

import json
import time
from typing import Any, Dict

from openai import OpenAI
from openai import (
  RateLimitError,
  APIConnectionError,
  APITimeoutError,
  InternalServerError,
)

client = OpenAI(timeout=30.0, max_retries=0)  # we control retries manually

RETRYABLE_ERRORS = (
  RateLimitError,
  APIConnectionError,
  APITimeoutError,
  InternalServerError,
)

def call_llm_with_schema(
    *,
    input_messages: list[dict],
    schema: dict,
    model: str = "gpt-4o-mini-2024-07-18",
    max_attempts: int = 3,
) -> Dict[str, Any]:
  backoff = 1.0

  for attempt in range(1, max_attempts + 1):
    try:
      resp = client.responses.create(
        model=model,
        input=input_messages,
        temperature=0.0,
        text={
          "format": {
            "type": "json_schema",
            "name": "output",
            "schema": schema,
            "strict": True,
          }
        },
      )

      print(
        {
          "response_id": resp.id,
          "request_id": getattr(resp, "_request_id", None),
          "usage": resp.usage,
        }
      )

      # Parse JSON
      data = json.loads(resp.output_text)
      return data

    except RETRYABLE_ERRORS as e:
      if attempt == max_attempts:
        raise

      time.sleep(backoff)
      backoff *= 2

    except json.JSONDecodeError as e:
      # Model violated schema or output was truncated
      raise RuntimeError("Model returned invalid JSON") from e

  raise RuntimeError("Unreachable")

# Example usage
name = "Harun Isik"
age = 38

try:
  result = call_llm_with_schema(
    input_messages=[
      {
        "role": "user",
        "content": "Provide your name and age."
      }
    ],
    schema={
      "type": "object",
      "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0}
      },
      "required": ["name", "age"],
      "additionalProperties": False
    }
  )
except RuntimeError:
  result = {
    "name": name,
    "age": age,
    "feedback": "Sorry — feedback is unavailable right now."
  }

print("Final result:", result)