from typing import TypedDict

from langgraph.graph import StateGraph, END
from app.rag.retriever import get_retriever


class AgentState(TypedDict):
    question: str
    context: str


def retrieve(state: AgentState):
    retriever = get_retriever()
    documents = retriever.invoke(state["question"])

    context = "\n\n".join(doc.page_content for doc in documents)

    return {
        "context": context
    }


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", END)

    return workflow.compile()
