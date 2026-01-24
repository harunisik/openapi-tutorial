"""
Running example: explicit state/memory in a *manual agent loop* (no magic).
- Uses LangChain ChatOpenAI
- Uses a tool (@tool)
- Stores state across steps: messages + scratchpad (tool calls/observations)
- Demonstrates bounded memory (keeps only last N steps)

Prereqs:
  pip install -U langchain-core langchain-openai python-dotenv pydantic
  export OPENAI_API_KEY=...

Run:
  python agent_state_demo.py
"""

from __future__ import annotations

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
# from langchain_core.messages import Humanaive:  # type: ignore
# If your editor complains, remove this block; it's here only for type hints in some IDEs.
# ...
# except Exception:
# pass
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    AIMessage,
)
from langchain_core.tools import tool


# ----------------------------
# 1) Tools
# ----------------------------

mock_db = {
    "A100": {"order_id": "A100", "status": "SHIPPED", "eta_days": 2},
    "B200": {"order_id": "B200", "status": "PROCESSING", "eta_days": 5},
    "C300": {"order_id": "C300", "status": "DELIVERED", "eta_days": 0},
}

@tool
def get_order_status(order_id: str) -> dict:
    """Look up an order status by order_id. Returns a JSON-like dict."""
    # Pretend this is a real API call / DB query
    return mock_db.get(order_id, {"order_id": order_id, "status": "NOT_FOUND", "eta_days": None})

@tool
def cancel_order(order_id: str) -> dict:
    """Cancel an order by order_id. Returns a JSON-like dict."""
    # Pretend this is a real API call / DB query
    if order_id in mock_db:
        current_status = mock_db[order_id]["status"]
        mock_db[order_id]["status"] = "CANCELLED"
        return {"order_id": order_id, "status": "CANCELLED", "previous_status": current_status}
    else:
        return {"order_id": order_id, "status": "NOT_FOUND"}

TOOLS = {get_order_status.name: get_order_status, cancel_order.name: cancel_order}


# ----------------------------
# 2) Agent "decision" schema
# ----------------------------

class ToolCall(BaseModel):
    name: Literal["get_order_status", "cancel_order"]
    arguments: str


class AgentDecision(BaseModel):
    """What the model must decide each loop."""
    action: Literal["tool", "final"] = Field(..., description="tool = call a tool, final = answer user")
    tool_call: Optional[ToolCall] = None
    final_answer: Optional[str] = None


# ----------------------------
# 3) Prompts
# ----------------------------

SYSTEM = """You are a careful customer-support agent.

You must:
- Use the tool if you need order status.
- Never invent an order status.
- Keep the answer short and helpful.

State you have:
- conversation_messages: the chat so far
- scratchpad_steps: previous tool calls and observations (internal)

Decide the next action:
- If you need order status: action="tool" with tool_call
- If you can answer now: action="final" with final_answer

Return ONLY a JSON object matching the required schema.
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", SYSTEM),
    # We pass the state in explicitly (this is the "memory")
    ("human", "Conversation so far:\n{conversation}\n\n"
              "Scratchpad:\n{scratchpad}\n\n"
              "User says:\n{user_input}")
])


# ----------------------------
# 4) Utilities to render state
# ----------------------------

def render_conversation(messages: List[BaseMessage], keep_last: int = 8) -> str:
    """Bounded 'conversation memory' to control token growth."""
    trimmed = messages[-keep_last:]
    lines = []
    for m in trimmed:
        role = "user" if isinstance(m, HumanMessage) else ("assistant" if isinstance(m, AIMessage) else "system")
        lines.append(f"{role}: {m.content}")
    return "\n".join(lines)


def render_scratchpad(steps: List[Dict[str, Any]], keep_last: int = 6) -> str:
    """Bounded 'agent scratchpad' memory."""
    trimmed = steps[-keep_last:]
    return json.dumps(trimmed, indent=2, ensure_ascii=False)


# ----------------------------
# 5) The agent loop (explicit state)
# ----------------------------

def run_agent_turn(state: Dict[str, Any], user_input: str, *, max_steps: int = 4) -> str:
    """
    State shape:
      state = {
        "messages": [SystemMessage|HumanMessage|AIMessage...],
        "scratchpad_steps": [ {"tool": "...", "args": {...}, "observation": {...}}, ...]
      }
    """
    llm = ChatOpenAI(model="gpt-4.1-mini", temperature=0).with_structured_output(AgentDecision)

    # Add the new user message to conversation memory
    state["messages"].append(HumanMessage(content=user_input))

    for step_idx in range(max_steps):
        conversation_text = render_conversation(state["messages"])
        scratchpad_text = render_scratchpad(state["scratchpad_steps"])

        decision: AgentDecision = (prompt | llm).invoke({
            "conversation": conversation_text,
            "scratchpad": scratchpad_text,
            "user_input": user_input,
        })

        print("\n--- Step", step_idx + 1, "---")
        print(decision.action)

        if decision.action == "final":
            answer = decision.final_answer or "I’m not sure. Could you clarify?"
            state["messages"].append(AIMessage(content=answer))
            return answer

        # action == "tool"
        if not decision.tool_call:
            # Guardrail: if model chose tool but didn't specify it, stop safely
            answer = "I couldn’t determine the right tool call. Which order ID should I check?"
            state["messages"].append(AIMessage(content=answer))
            return answer

        print("Tool call:", decision.tool_call)
        print("---------------")

        tool_name = decision.tool_call.name
        tool_args = decision.tool_call.arguments

        tool_fn = TOOLS.get(tool_name)
        if not tool_fn:
            answer = f"I don’t have access to the tool '{tool_name}'."
            state["messages"].append(AIMessage(content=answer))
            return answer

        observation = tool_fn.invoke(json.loads(tool_args)["order_id"])

        # Store in scratchpad (this is "agent memory/state in the loop")
        state["scratchpad_steps"].append({
            "tool": tool_name,
            "args": tool_args,
            "observation": observation,
        })

        # Also store a short assistant message reflecting the observation (optional)
        state["messages"].append(AIMessage(content=f"[Tool {tool_name} executed]"))

    # If we hit step limit, stop safely
    answer = "I hit my step limit while checking that. What’s the order ID and any extra context?"
    state["messages"].append(AIMessage(content=answer))
    return answer


# ----------------------------
# 6) Demo
# ----------------------------

def main():
    # Initialize state (this is your memory container)
    state: Dict[str, Any] = {
        "messages": [
            SystemMessage(content="You are a customer support chat."),
        ],
        "scratchpad_steps": [],
    }

    while True:
        user_question = input("Question (press Enter to quit): ").strip()
        if not user_question:
            print("Exiting...")
            break
        result = run_agent_turn(state, user_question)
        print("Assistant:", result)

    # print("\n--- Turn 1 ---")
    # print("Assistant:", run_agent_turn(state, "Hi—what’s the status of order A100?"))
    #
    # print("\n--- Turn 2 (uses memory) ---")
    # print("Assistant:", run_agent_turn(state, "And what about B200?"))
    #
    # print("\n--- Turn 3 (no tool needed) ---")
    # print("Assistant:", run_agent_turn(state, "Thanks! If A100 is shipped, when should it arrive?"))
    #
    # print("\n--- Internal state (for learning) ---")
    # print("Conversation memory:\n", render_conversation(state["messages"], keep_last=50))
    # print("\nScratchpad memory:\n", render_scratchpad(state["scratchpad_steps"], keep_last=50))
