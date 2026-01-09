from __future__ import annotations

import json
import time
from typing import Any, Dict, List, Tuple

from openai import OpenAI
from openai import (
  RateLimitError,
  APIConnectionError,
  APITimeoutError,
  InternalServerError,
)

RETRYABLE = (RateLimitError, APIConnectionError, APITimeoutError, InternalServerError)

CLIENT = OpenAI(timeout=30.0, max_retries=0)


EXTRACTION_SCHEMA: Dict[str, Any] = {
  "type": "object",
  "properties": {
    "customer_name": {"type": "string"},
    "email": {"type": "string"},
    "issue_summary": {"type": "string"},
    "urgency": {"type": "string", "enum": ["low", "medium", "high"]},
    "product_area": {"type": "string", "enum": ["billing", "export", "login", "performance", "other"]},
  },
  "required": ["customer_name", "email", "issue_summary", "urgency", "product_area"],
  "additionalProperties": False,
}


def extract_ticket(text: str, *, max_attempts: int = 3) -> Tuple[Dict[str, Any], Dict[str, Any]]:
  """
  Extract a structured support ticket from freeform text.

  Returns:
      (data, telemetry)
      - data: validated JSON (schema enforced by model)
      - telemetry: response_id, request_id, usage token counts
  """
  backoff = 1.0
  last_err: Exception | None = None

  instructions = (
    "You are a data extraction service.\n"
    "Return ONLY valid JSON that matches the provided schema.\n"
    "Do not include extra keys or any explanation."
  )

  for attempt in range(1, max_attempts + 1):
    try:
      resp = CLIENT.responses.create(
        model="gpt-4o-mini-2024-07-18",
        instructions=instructions,
        input=f"Text:\n{text}",
        temperature=0.0,
        max_output_tokens=200,
        text={
          "format": {
            "type": "json_schema",
            "name": "support_ticket",
            "schema": EXTRACTION_SCHEMA,
            "strict": True,
          }
        },
      )

      data = json.loads(resp.output_text)

      telemetry = {
        "response_id": resp.id,
        "request_id": getattr(resp, "_request_id", None),
        "usage": {
          "input_tokens": resp.usage.input_tokens,
          "output_tokens": resp.usage.output_tokens,
          "total_tokens": resp.usage.total_tokens,
        },
      }
      return data, telemetry

    except RETRYABLE as e:
      last_err = e
      if attempt == max_attempts:
        break
      time.sleep(backoff)
      backoff *= 2

    except json.JSONDecodeError as e:
      # Schema enforcement should prevent this most of the time,
      # but truncation or upstream issues can still break JSON.
      raise RuntimeError("Invalid JSON returned by model") from e

  raise RuntimeError(f"LLM request failed after {max_attempts} attempts: {last_err}")
