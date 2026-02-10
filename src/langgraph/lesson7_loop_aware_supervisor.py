from typing import TypedDict

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
    research_notes: str

    draft_answer_a: str
    draft_answer_b: str

    final_answer: str
    validation_feedback: str

    retry_count: int
    max_retries: int


def research_node(state: AgentState) -> dict:
    # print(f"Supervisor state: {state}")
    # print("----------------------------------")

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


def reasoning_node_a(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(
            content="You are Reasoner A. Produce a clear, structured answer."
        ),
        HumanMessage(
            content=f"Research:\n{state['research_notes']}\n\nQuestion:\n{state['user_query']}"
        )
    ])
    return {"draft_answer_a": response.content}

def reasoning_node_b(state: AgentState) -> dict:
    response = llm.invoke([
        SystemMessage(
            content="You are Reasoner B. Provide an alternative reasoning approach."
        ),
        HumanMessage(
            content=f"Research:\n{state['research_notes']}\n\nQuestion:\n{state['user_query']}"
        )
    ])
    return {"draft_answer_b": response.content}


def reasoning_node(state: AgentState) -> dict:
    # print(f"Supervisor state: {state}")
    # print("----------------------------------")

    """Composite reasoning node that runs both Reasoner A and Reasoner B.

    This keeps the supervisor logic unchanged (it can still route to the
    "reasoning" node) while ensuring both drafts are produced for validation.
    Reasoners must run independently (no sharing of each other's drafts).
    """
    # Run Reasoner A (does NOT expose its draft to Reasoner B)
    out_a = reasoning_node_a(state)

    # Run Reasoner B with the original state only — do NOT merge out_a into the state
    out_b = reasoning_node_b(state)

    # Return combined outputs so the graph state contains both drafts for the validator
    return {**out_a, **out_b}


def validation_node(state: AgentState) -> dict:
    # print(f"Supervisor state: {state}")
    # print("----------------------------------")

    # Ask the judge to explicitly pick A, B, or REJECT for easier parsing
    response = llm.invoke([
        SystemMessage(
            content=(
                "You are a strict judge. Compare Answer A and Answer B. "
                "Reply with exactly one of: 'A' if Answer A is better, 'B' if Answer B is better, "
                "or 'REJECT' if neither is acceptable. After that, you may give a brief justification."
            )
        ),
        HumanMessage(
            content=(
                f"Answer A:\n{state.get('draft_answer_a', '')}\n\n"
                f"Answer B:\n{state.get('draft_answer_b', '')}"
            )
        )
    ])

    verdict_raw = response.content.strip()
    verdict = verdict_raw.lower()

    if "reject" in verdict:
        return {
            "validation_feedback": response.content
        }

    return {
        "final_answer": response.content,
        "validation_feedback": response.content
    }



def supervisor_node(state: AgentState) -> dict:
    print_state(state)

    if not state.get("research_notes"):
        return {"next_step": "research"}

    if not state.get("draft_answer_a") or not state.get("draft_answer_b"):
        return {"next_step": "reasoning"}

    if not state.get("final_answer"):
        if state["retry_count"] < state["max_retries"]:
            return {
                "retry_count": state["retry_count"] + 1,
                "next_step": "reasoning"
            }

    return {"next_step": "end"}



def main():
    print("Hello from lesson 6! This is where we'll implement the graph node class.")
    graph = StateGraph(AgentState)

    graph.add_node("supervisor", supervisor_node)
    graph.add_node("research", research_node)
    # keep the single "reasoning" node name but wire it to the composite that runs both A and B
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
    graph.add_edge("validation", "supervisor")
    graph.add_edge("reasoning", "validation")

    app = graph.compile()
    result = app.invoke({
        "user_query": "Explain how CRISPR gene editing works",
        "retry_count": 0,
        "max_retries": 2,
    })

    print(result["final_answer"])

def print_state(state: AgentState):
    print(f"Current state: "
          f"user_query: {str(state.get('user_query', ''))[:20]}, "
          f"research_notes: {str(state.get('research_notes', ''))[:20]}, "
          f"draft_answer_a: {str(state.get('draft_answer_a', ''))[:20]}, "
          f"draft_answer_b: {str(state.get('draft_answer_b', ''))[:20]}, "
          f"final_answer: {str(state.get('final_answer', ''))[:50]},) "
          f"validation_feedback: {str(state.get('validation_feedback', ''))[:50]},) "
          f"retry_count: {state.get('retry_count', 0)}, "
          f"max_retries: {state.get('max_retries', 0)}")
    print("----------------------------------")