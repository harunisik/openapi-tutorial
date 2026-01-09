from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI

# Create the streaming LLM
llm = ChatOpenAI(
    model="gpt-4.1-mini",
    temperature=0,
    streaming=True
)

# Helper: try to get a tiktoken encoder for the model, else None
def _get_encoder_for_model(model_name):
    try:
        import tiktoken
        return tiktoken.encoding_for_model(model_name)
    except Exception:
        return None

# Stream prompt and stop after max_tokens (best-effort: exact if tiktoken available)
def stream_with_token_limit(prompt: list[SystemMessage | HumanMessage], max_tokens: int):
    encoder = _get_encoder_for_model(llm.model_name)
    remaining = max_tokens

    for chunk in llm.stream(prompt):
        text = chunk.content or ""
        if not text:
            continue

        if encoder is not None:
            # precise token slicing using tiktoken
            token_ids = encoder.encode(text)
            total_len = len(token_ids)
            if total_len <= remaining:
                print(text, end="", flush=True)
                remaining -= total_len
            else:
                # print only the token-limited portion
                partial = encoder.decode(token_ids[:remaining])
                print(partial, end="", flush=True)
                break
        else:
            # fallback: approximate by words
            words = text.split()
            total_len = len(words)
            if total_len <= remaining:
                print(text, end="", flush=True)
                remaining -= total_len
            else:
                partial = " ".join(words[:remaining])
                print(partial, end="", flush=True)
                break

    print()  # ensure newline after streaming

def main():
    prompt = "Explain embeddings simply and in detail."  # example long prompt
    max_tokens = 50  # stop after 50 tokens (adjust as needed)
    stream_with_token_limit([
        SystemMessage(content="You are a helpful assistant."),
        HumanMessage(content=prompt)
    ], max_tokens)
