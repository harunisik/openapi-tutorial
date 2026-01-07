from openai import OpenAI
import math

DOCUMENTS = [
    {
        "id": "doc1",
        "text": "Python virtual environments isolate dependencies using venv."
    },
    {
        "id": "doc2",
        "text": "Pandas is a Python library used for data analysis."
    },
    {
        "id": "doc3",
        "text": "Java uses the JVM to run code across platforms."
    },
]


client = OpenAI()

doc_embeddings = []

for doc in DOCUMENTS:
    emb = client.embeddings.create(
        model="text-embedding-3-small",
        input=doc["text"],
    ).data[0].embedding

    doc_embeddings.append(
        {
            "id": doc["id"],
            "text": doc["text"],
            "embedding": emb,
        }
    )

query = "How do I isolate Python dependencies?"

query_embedding = client.embeddings.create(
    model="text-embedding-3-small",
    input=query,
).data[0].embedding


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = math.sqrt(sum(x * x for x in a))
    norm_b = math.sqrt(sum(y * y for y in b))
    return dot / (norm_a * norm_b)

scored_docs = []

for doc in doc_embeddings:
    score = cosine_similarity(query_embedding, doc["embedding"])
    scored_docs.append((score, doc))

scored_docs.sort(reverse=True)

top_docs = scored_docs[:2]

context = "\n\n".join(
    f"[{doc['id']}] {doc['text']}"
    for _, doc in top_docs
)

instructions = (
    "You are a factual assistant.\n"
    "Answer the question using ONLY the provided context.\n"
    "If the answer is not in the context, say you don't know."
)

input_text = (
    f"Context:\n{context}\n\n"
    f"Question: {query}"
)

resp = client.responses.create(
    model="gpt-4o-mini",
    instructions=instructions,
    input=input_text,
    temperature=0.0,
    max_output_tokens=150,
)

print(resp.output_text)
