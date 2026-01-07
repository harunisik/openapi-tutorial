from __future__ import annotations

import math
from openai import OpenAI

client = OpenAI()

DOCS = [
  {"id": "policy_1", "text": "Refunds are allowed within 14 days with a receipt."},
  {"id": "policy_2", "text": "Enterprise plans include priority support with a 4-hour SLA."},
  {"id": "policy_3", "text": "Export feature is available on Pro and Enterprise tiers."},
]

def cosine_similarity(a, b) -> float:
  dot = sum(x * y for x, y in zip(a, b))
  norm_a = math.sqrt(sum(x * x for x in a))
  norm_b = math.sqrt(sum(y * y for y in b))
  return dot / (norm_a * norm_b)

def embed(text: str) -> list[float]:
  return client.embeddings.create(
    model="text-embedding-3-small",
    input=text,
  ).data[0].embedding

def retrieve(query: str, top_k: int = 2):
  q = embed(query)
  scored = []
  for d in DOCS:
    s = cosine_similarity(q, embed(d["text"]))
    scored.append((s, d))
  scored.sort(reverse=True)
  return [d for _, d in scored[:top_k]]

def answer(query: str) -> str:
  top = retrieve(query, top_k=2)
  context = "\n".join([f"[{d['id']}] {d['text']}" for d in top])

  instructions = (
    "You are a factual assistant.\n"
    "Use ONLY the provided context.\n"
    "If the answer is missing, say \"I don't know\".\n"
    "Cite sources using the bracketed ids, e.g. [policy_1]."
  )

  resp = client.responses.create(
    model="gpt-4o-mini",
    instructions=instructions,
    input=f"Context:\n{context}\n\nQuestion: {query}",
    temperature=0.0,
    max_output_tokens=200,
  )
  return resp.output_text

if __name__ == "__main__":
  q = "Which plans have export?"
  print(answer(q))
