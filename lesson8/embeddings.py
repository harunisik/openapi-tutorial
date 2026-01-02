from openai import OpenAI

client = OpenAI()

resp = client.embeddings.create(
  model="text-embedding-3-small",
  input="Python virtual environments isolate dependencies.",
)

vector = resp.data[0].embedding
print(len(vector))  # e.g., 1536
print(vector[:8])   # print first 5 dimensions