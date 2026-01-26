from __future__ import annotations

import os
import json
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from langchain_openai import ChatOpenAI
from langchain_core.tools import tool
from langchain_core.messages import (
    BaseMessage,
    HumanMessage,
    SystemMessage,
    ToolMessage,
)
from langchain_core.runnables import RunnableLambda

from src.config import OPENAI_MODEL


# -----------------------------
# 1) Minimal "backend" state
# -----------------------------
@dataclass
class CartStore:
    items: Dict[str, int] = field(default_factory=dict)

    def add(self, sku: str, qty: int) -> None:
        if qty <= 0:
            raise ValueError("qty must be > 0")
        self.items[sku] = self.items.get(sku, 0) + qty

    def view(self) -> Dict[str, int]:
        return dict(self.items)


CATALOG = [
    {"sku": "TSHIRT-001", "name": "Plain T-Shirt", "price_gbp": 12.99},
    {"sku": "MUG-042", "name": "Ceramic Mug", "price_gbp": 8.50},
    {"sku": "HOODIE-123", "name": "Zip Hoodie", "price_gbp": 39.00},
]


# -----------------------------
# 2) Tools
# -----------------------------
class SearchInput(BaseModel):
    query: str = Field(..., description="Free-text product search query")


@tool(args_schema=SearchInput)
def search_catalog(query: str) -> List[dict]:
    """Search the product catalog by name keyword; returns matching products."""
    q = query.lower().strip()
    return [p for p in CATALOG if q in p["name"].lower() or q in p["sku"].lower()]


class GetProductDetailsInput(BaseModel):
    sku: str = Field(..., description="Product SKU, e.g. 'MUG-042'")


@tool(args_schema=GetProductDetailsInput)
def get_product_details(sku: str) -> Optional[dict]:
    """Get product details by SKU."""
    for p in CATALOG:
        if p["sku"] == sku:
            return p
    raise ValueError(f"SKU {sku} not found")


class AddToCartInput(BaseModel):
    sku: str = Field(..., description="Product SKU, e.g. 'MUG-042'")
    qty: int = Field(..., ge=1, description="Quantity to add (>=1)")


def make_add_to_cart_tool(cart: CartStore):
    @tool(args_schema=AddToCartInput)
    def add_to_cart(sku: str, qty: int) -> str:
        """Add a product to the cart."""
        cart.add(sku=sku, qty=qty)
        return f"Added {qty} x {sku} to cart."
    return add_to_cart


def make_view_cart_tool(cart: CartStore):
    @tool
    def view_cart() -> dict:
        """View current cart contents."""
        return cart.view()
    return view_cart


# -----------------------------
# 3) LCEL "planner" runnable
# -----------------------------
SYSTEM = SystemMessage(
    content=(
        "You are ShopAgent, a careful e-commerce assistant.\n"
        "Use tools to search products and manage the cart.\n"
        "Rules:\n"
        "- If you need product info, call search_catalog.\n"
        "- If user wants product details, call get_product_details (needs SKU).\n"
        "- If user wants to add items, confirm SKU and quantity, then call add_to_cart.\n"
        "- If user asks what's in the cart, call view_cart.\n"
        "- If you don't know the SKU, search first.\n"
        "Be concise and confirm actions.\n"
    )
)


def build_planner(tools):
    """
    Returns an LCEL runnable that takes {"messages": [..]} and returns an AI message
    that may include tool_calls.
    """
    llm = ChatOpenAI(
        model=os.getenv("OPENAI_MODEL", OPENAI_MODEL),
        temperature=0,
    ).bind_tools(tools)

    def _invoke(inputs: Dict[str, Any]):
        msgs: List[BaseMessage] = [SYSTEM] + inputs["messages"]
        return llm.invoke(msgs)

    return RunnableLambda(_invoke)


# -----------------------------
# 4) Tool execution
# -----------------------------
def run_tool_calls(ai_message, tools_by_name: Dict[str, Any]) -> List[ToolMessage]:
    tool_messages: List[ToolMessage] = []
    for call in getattr(ai_message, "tool_calls", []) or []:
        name = call["name"]
        args = call.get("args", {})  # already parsed JSON
        tool = tools_by_name[name]
        try:
            obs = tool.invoke(args)
            content = obs if isinstance(obs, str) else json.dumps(obs)
        except Exception as e:
            # Give the model something actionable to recover with.
            content = f"TOOL_ERROR: {type(e).__name__}: {e}"
        tool_messages.append(ToolMessage(content=content, tool_call_id=call["id"]))
    return tool_messages


# -----------------------------
# 5) Minimal agent loop (no LangGraph, no AgentExecutor)
# -----------------------------
def chat_turn(
        user_text: str,
        messages: List[BaseMessage],
        planner,
        tools_by_name: Dict[str, Any],
        max_steps: int = 6,
) -> List[BaseMessage]:
    """
    Runs a single user turn with a controlled loop:
    plan (LLM) -> (optional) tool exec -> observe -> plan ...
    Stops when the LLM returns no tool_calls.
    """
    messages = messages + [HumanMessage(content=user_text)]

    for _ in range(max_steps):
        ai = planner.invoke({"messages": messages})
        messages = messages + [ai]

        if not getattr(ai, "tool_calls", None):
            # final answer for this turn
            return messages

        tool_msgs = run_tool_calls(ai, tools_by_name)
        messages = messages + tool_msgs

    # Safety fallback: if we hit max_steps, end the turn
    messages = messages + [
        ToolMessage(
            content="Max steps reached; stopping to avoid looping.",
            tool_call_id="loop_guard",
        )
    ]
    return messages


def main():
    cart = CartStore()
    tools = [search_catalog, make_add_to_cart_tool(cart), make_view_cart_tool(cart), get_product_details]
    tools_by_name = {t.name: t for t in tools}
    planner = build_planner(tools)

    print("ShopAgent (LCEL + tool calling) ready.")
    print("Try: 'Find a mug and add 2 to my cart' or 'What's in my cart?'\n")

    history: List[BaseMessage] = []

    while True:
        text = input("You> ").strip()
        if text.lower() in {"quit", "exit"}:
            break

        history = chat_turn(
            user_text=text,
            messages=history,
            planner=planner,
            tools_by_name=tools_by_name,
            max_steps=6,
        )

        # last AI message is typically the final answer
        last_ai: Optional[BaseMessage] = next(
            (m for m in reversed(history) if m.type == "ai"), None
        )
        print(f"\nAgent> {getattr(last_ai, 'content', '')}\n")
