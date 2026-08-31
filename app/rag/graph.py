from typing import TypedDict

from langgraph.graph import StateGraph, END

from app.rag.retriever import get_retriever
from app.rag.generator import generate_answer


class AgentState(TypedDict):
    question: str
    context: str
    answer: str


def retrieve(state: AgentState):
    retriever = get_retriever()
    documents = retriever.invoke(state["question"])

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    return {"context": context}


def generate(state: AgentState):
    answer = generate_answer(
        state["question"],
        state["context"]
    )

    return {"answer": answer}


def build_graph():
    workflow = StateGraph(AgentState)

    workflow.add_node("retrieve", retrieve)
    workflow.add_node("generate", generate)

    workflow.set_entry_point("retrieve")
    workflow.add_edge("retrieve", "generate")
    workflow.add_edge("generate", END)

    return workflow.compile()


graph = build_graph()


def ask_question(question: str):
    result = graph.invoke({
        "question": question,
        "context": "",
        "answer": ""
    })

    return result["answer"]
