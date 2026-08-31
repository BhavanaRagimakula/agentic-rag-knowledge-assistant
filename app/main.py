from fastapi import FastAPI
from pydantic import BaseModel

from app.rag.graph import ask_question

app = FastAPI(
    title="Agentic RAG Knowledge Assistant",
    description="AI-powered knowledge assistant using LangChain and LangGraph",
    version="1.0.0"
)


class QuestionRequest(BaseModel):
    question: str


@app.get("/")
def root():
    return {
        "message": "Agentic RAG Knowledge Assistant is running"
    }


@app.post("/ask")
def ask(request: QuestionRequest):
    answer = ask_question(request.question)

    return {
        "question": request.question,
        "answer": answer
    }
