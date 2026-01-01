from __future__ import annotations

import json
import time
from typing import Any, Dict, Callable, Optional

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
    fallback_factory: Optional[Callable[[BaseException], Dict[str, Any]]] = None,
) -> Dict[str, Any]:
  backoff = 1.0
  last_error: Optional[BaseException] = None

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

      data = json.loads(resp.output_text)
      return data

    except RETRYABLE_ERRORS as e:
      last_error = e
      if attempt == max_attempts:
        break
      time.sleep(backoff)
      backoff *= 2

    except json.JSONDecodeError as e:
      last_error = RuntimeError("Model returned invalid JSON")
      break

  if fallback_factory is not None and last_error is not None:
    return fallback_factory(last_error)

  if last_error is not None:
    raise last_error

  raise RuntimeError("Unreachable")


def fallback_factory(exc: BaseException) -> Dict[str, Any]:
  return {
    "name": "Unknown",
    "age": -1,
    "feedback": "Sorry \u2014 feedback is unavailable right now.",
    "error_type": type(exc).__name__,
  }


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
  },
  fallback_factory=fallback_factory,
)

print("Final result:", result)
