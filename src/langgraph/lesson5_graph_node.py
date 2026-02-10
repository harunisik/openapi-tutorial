from typing import TypedDict, Optional

from langchain_core.messages import SystemMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.constants import END
from langgraph.graph import StateGraph

from src.config import OPENAI_MODEL

llm = ChatOpenAI(
    model=OPENAI_MODEL,
    temperature=0.2
)

class AgentState(TypedDict):
    user_query: str
    research_notes: Optional[str]
    draft_answer: Optional[str]
    validation_feedback: Optional[str]
    final_answer: Optional[str]


def research_node(state: AgentState) -> dict:
    messages = [
        SystemMessage(
            content=(
                "You are a research agent. "
                "Gather factual, relevant information only. "
                "Do NOT answer the question."
            )
        ),
        HumanMessage(content=state["user_query"])
    ]

    response = llm.invoke(messages)

    return {
        "research_notes": response.content
    }

def reasoning_node(state: AgentState) -> dict:
    messages = [
        SystemMessage(
            content=(
                "You are a reasoning agent. "
                "Using the provided research notes, "
                "produce a clear, structured draft answer. "
                "Do NOT introduce new facts."
            )
        ),
        HumanMessage(
            content=(
                f"Question:\n{state['user_query']}\n\n"
                f"Research Notes:\n{state['research_notes']}"
            )
        )
    ]

    response = llm.invoke(messages)

    return {
        "draft_answer": response.content
    }

def validation_node(state: AgentState) -> dict:
    messages = [
        SystemMessage(
            content=(
                "You are a strict validator. "
                "Check the draft answer for correctness, completeness, "
                "and alignment with the research notes. "
                "If valid, approve it. If not, explain why."
            )
        ),
        HumanMessage(
            content=(
                f"Research Notes:\n{state['research_notes']}\n\n"
                f"Draft Answer:\n{state['draft_answer']}"
            )
        )
    ]

    response = llm.invoke(messages)

    feedback = response.content.lower()

    if "approve" in feedback or "valid" in feedback:
        return {
            "final_answer": state["draft_answer"],
            "validation_feedback": response.content
        }

    return {
        "validation_feedback": response.content
    }

def supervisor_node(state: AgentState) -> dict:
    if not state.get("research_notes"):
        return {"next_step": "research"}
    if not state.get("draft_answer"):
        return {"next_step": "reasoning"}
    if not state.get("final_answer"):
        return {"next_step": "validation"}
    return {"next_step": "end"}


def main():
    print("Hello from lesson 6! This is where we'll implement the graph node class.")
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    graph.add_node("reasoning", reasoning_node)
    graph.add_node("validation", validation_node)

    graph.set_entry_point("supervisor")

    graph.add_conditional_edges(
        "supervisor",
        lambda s: s["next_step"],
        {
            "research": "research",
            "reasoning": "reasoning",
            "validation": "validation",
            "end": END,
        }
    )

    graph.add_edge("research", "supervisor")
    graph.add_edge("reasoning", "supervisor")
    graph.add_edge("validation", "supervisor")

    app = graph.compile()
    result = app.invoke({
        "user_query": "Explain how CRISPR gene editing works"
    })

    print(result["final_answer"])
