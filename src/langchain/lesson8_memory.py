from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda, RunnablePassthrough, RunnableMap
from langchain_openai import ChatOpenAI
import os

memory_enabled = os.getenv("MEMORY_ENABLED", "true").lower() == "true"

llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0)

prompt = ChatPromptTemplate.from_messages([
    ("system", "You explain concepts clearly."),
    ("system", "Memory: {memory}"),
    ("user", "Question: {question}")
])

# Summarization prompt (keeps summaries short and factual)
summary_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a concise conversation summarizer."),
    ("user",
     "Summarize the following conversation into 2-3 short sentences, keeping key facts and decisions:\n\n{memory}")
])

memory: list[str] = []

summary_chain = summary_prompt | llm | StrOutputParser()

def summarize_history_text(history_text: str) -> str:
    """Run a small chain to summarize the raw history text."""
    # invoke with the memory text and return the parsed string
    return summary_chain.invoke({"memory": history_text})


def summarize_history() -> str:
    """Build raw history and return a short summary."""
    if not memory_enabled:
        return ""  # memory disabled -> inject empty string
    raw = "\n".join(memory)
    if not raw.strip():
        return ""  # empty memory -> inject empty string
    return summarize_history_text(raw)


# Runnable that will inject the summarized history into the main chain
history_runnable = RunnableLambda(lambda _: summarize_history())

inputs = RunnableMap({
    "memory": history_runnable,
    "question": lambda x: x["question"]
})

chain = (inputs
    | RunnableLambda(lambda x: (print(f"\n-------\ninputs: {x}\n-----\n"), x)[1])
    | prompt | llm | StrOutputParser())


def main():
    while True:
        user_question = input("Question (press Enter to quit): ").strip()
        if not user_question:
            print("Exiting...")
            break
        result = chain.invoke({"question": user_question})
        if memory_enabled:
            # keep raw exchange in memory, summarizer will compress on next turn
            memory.append(f"User: {user_question}\nAI: {result}")
        print(result)

# 2) Architectural improvements (cleaner LCEL + less surprise)
# ✅ A) Don’t recreate summary_chain on every call
# Right now summarize_history_text builds:
# summary_chain = summary_prompt | llm | StrOutputParser()
# every time. Build it once globally.
#
# ✅ B) Add a “memory summarization policy”
# Common pattern:
# Keep last N turns raw
# Keep one running summary
# Only re-summarize when raw turns exceed N or token threshold
# Your approach (summarize whole raw transcript every turn) works but becomes expensive as memory grows.
#
# ✅ C) Avoid feeding summaries back into the same model without guardrails
# Summaries can drift (“telephone game”). In production you’d:
# store raw messages + summary separately
# periodically refresh summary from raw, or use a stricter summarizer prompt
#                                                                     or keep short “facts” list rather than narrative summary
#
# 3) Production/performance considerations
# Latency
# You’re doing two LLM calls per user question when memory exists:
# summarize history
# answer question
# That doubles latency/cost.
# Optimization options:
# Summarize every N turns instead of every turn
# Summarize asynchronously (in real systems via background job/queue)
# Use a cheaper model for summarization (often a smaller/cheaper model)
#
# Prompt correctness
# For RAG later, you’ll want:
#     memory summary in system
# retrieved context in user
# strict grounding rules