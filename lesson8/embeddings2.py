from openai import OpenAI
import math

client = OpenAI()

def cosine_similarity(a: list[float], b: list[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

documents = [
    "Python supports virtual environments using venv.",
    "Java uses the JVM for cross-platform execution.",
    "Pandas is a Python library for data analysis.",
]

doc_embeddings = []

for doc in documents:
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc,
    ).data[0].embedding

    doc_embeddings.append(emb)


query = "How do I isolate Python dependencies?"

query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query,
).data[0].embedding

scores = []

for doc, emb in zip(documents, doc_embeddings):
    score = cosine_similarity(query_embedding, emb)
    scores.append((score, doc))

scores.sort(reverse=True)

for score, doc in scores:
    print(f"{score:.3f} | {doc}")
