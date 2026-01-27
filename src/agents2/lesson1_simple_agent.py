from __future__ import annotations

from typing import Any, Dict, List

from langchain_core.messages import (
    AIMessage,
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from src.config import OPENAI_MODEL


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


# -------------------------
# 3) State + loop primitives
# -------------------------
def _init_state(user_input: str) -> Dict[str, Any]:
    # State is explicit and easy to persist/log
    messages: List[BaseMessage] = [
        SystemMessage(content="You are a precise assistant. Use tools when needed."),
        HumanMessage(content=user_input),
    ]
    return {"messages": messages}


def _plan(state: Dict[str, Any]) -> Dict[str, Any]:
    """LLM step: decides next action (tool call) or returns final answer."""
    ai: AIMessage = llm.invoke(state["messages"])
    return {"messages": state["messages"] + [ai]}


def _execute_tools(state: Dict[str, Any]) -> Dict[str, Any]:
    """Executor step: run any tool calls from the last AIMessage and append ToolMessage observations."""
    last = state["messages"][-1]
    if not isinstance(last, AIMessage) or not last.tool_calls:
        return state  # nothing to execute

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
                    content=f"ERROR: Unknown tool '{name}'. Available: {list(TOOL_BY_NAME)}",
                )
            )
            continue

        try:
            # Prefer tool.invoke for consistent behavior across tool types
            result = tool.invoke(args)
            tool_messages.append(ToolMessage(tool_call_id=call_id, content=str(result)))
        except Exception as e:
            tool_messages.append(ToolMessage(tool_call_id=call_id, content=f"ERROR: {e!r}"))

    return {"messages": state["messages"] + tool_messages}


def _should_continue(state: Dict[str, Any]) -> bool:
    """Stop when the last AI message has no tool calls (i.e., it produced a final answer)."""
    last = state["messages"][-1]
    return isinstance(last, AIMessage) and bool(last.tool_calls)


def run_agent(user_input: str, *, max_steps: int = 8) -> AIMessage:
    """
    Minimal agent loop:
    - PLAN: model proposes tool calls or final answer
    - EXECUTE: run tool calls, append ToolMessage observations
    - repeat until final answer or step limit
    """
    state = _init_state(user_input)

    for _ in range(max_steps):
        state = _plan(state)
        if not _should_continue(state):
            break
        state = _execute_tools(state)

    last = state["messages"][-1]
    if not isinstance(last, AIMessage):
        # If we ended on ToolMessage, do one last planning step for a final answer
        state = _plan(state)
        last = state["messages"][-1]

    assert isinstance(last, AIMessage)
    return last


# -------------------------
# 4) Demo
# -------------------------
def main() -> None:
    final = run_agent("Compute (7 * 9) + 10. Use tools.")
    print(final.content)
