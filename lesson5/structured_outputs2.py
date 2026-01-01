from __future__ import annotations

import json
from openai import OpenAI

from pydantic import BaseModel, ValidationError

class Person(BaseModel):
    name: str
    age: int
    feedback: str

client = OpenAI()

schema = {
    "type": "object",
    "properties": {
        "name": {"type": "string"},
        "age": {"type": "integer", "minimum": 0},
        "feedback": {"type": "string"},
    },
    "required": ["name", "age", "feedback"],
    "additionalProperties": False,
}

resp = client.responses.create(
    model="gpt-4o-mini-2024-07-18",
    input=[
        {"role": "system", "content": "Return structured data only."},
        {"role": "user", "content": "name=Alice\nage=30\nGive friendly feedback."},
    ],
    text={
        "format": {
            "type": "json_schema",
            "name": "person",
            "schema": schema,
            "strict": True,
        }
    },
    temperature=0.0,
)

try:
    data = json.loads(resp.output_text)
    print(data)
    person = Person(**data)
    print(person)
except ValidationError as e:
    print("Invalid output:", e)