from __future__ import annotations

from typing import Any, Dict, List, Optional

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.config import OPENAI_MODEL

SYSTEM = SystemMessage(content="You are a precise assistant. Use tools when needed.")


# -------------------------
# 1) Tools (action space)
# -------------------------
@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


@tool
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


TOOLS = [multiply, add]
TOOL_BY_NAME = {t.name: t for t in TOOLS}


# -------------------------
# 2) Model (policy)
# -------------------------
llm = ChatOpenAI(model=OPENAI_MODEL, temperature=0).bind_tools(TOOLS)
# bind_tools enables native tool-calling, and tool calls appear on AIMessage.tool_calls :contentReference[oaicite:1]{index=1}


def _plan(state: List[BaseMessage]) -> AIMessage:
    """LLM step: decides next action (tool call) or returns AI message."""
    ai: AIMessage = llm.invoke(state)
    return ai


def _execute_tools(last: AIMessage) -> List[ToolMessage]:
    """Executor step: run any tool calls from the last AIMessage and append ToolMessage observations."""

    tool_messages: List[ToolMessage] = []
    for call in last.tool_calls:
        name = call["name"]
        args = call.get("args", {})
        call_id = call.get("id")

        tool = TOOL_BY_NAME.get(name)
        if tool is None:
            # Observation for invalid tool name (guardrail)
            tool_messages.append(
                ToolMessage(
                    tool_call_id=call_id,
                    content=f"TOOL_ERROR: Unknown tool '{name}'. Available: {list(TOOL_BY_NAME)}",
                )
            )
            continue

        try:
            # Prefer tool.invoke for consistent behavior across tool types
            result = tool.invoke(args)
            tool_messages.append(ToolMessage(tool_call_id=call_id, content=str(result)))
        except Exception as e:
            tool_messages.append(ToolMessage(tool_call_id=call_id, content=f"TOOL_ERROR: {e!r}"))

    return tool_messages


def _should_continue(state: Dict[str, Any]) -> bool:
    """Stop when the last AI message has no tool calls (i.e., it produced a final answer)."""
    last = state["messages"][-1]
    return isinstance(last, AIMessage) and bool(last.tool_calls)


def run_agent(user_input: str, *, max_steps: int = 5) -> list[BaseMessage] | None:
    """
    Minimal agent loop:
    - PLAN: model proposes tool calls or final answer
    - EXECUTE: run tool calls, append ToolMessage observations
    - repeat until final answer or step limit
    """
    state: List[BaseMessage] = [SYSTEM, HumanMessage(content=user_input)]

    for _ in range(max_steps):
        ai = _plan(state)
        state.append(ai)

        # Stop if last AIMessage has no tool calls
        if not ai.tool_calls:
            return state

        observation = _execute_tools(ai)
        state.extend(observation)

    state.append(
        ToolMessage(
            content="Max steps reached; stopping to avoid looping. Please restate your request or try again.",
            tool_call_id="loop_guard",
        )
    )

    return state

# -------------------------
# 4) Demo
# -------------------------
def main() -> None:
    user_input = "Compute (7 * 9) + 10. Use tools."

    final = run_agent(user_input)
    # last AI message is typically the final answer
    last: Optional[BaseMessage] = final[-1]
    print(last.content)
