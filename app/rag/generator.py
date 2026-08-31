from langchain_openai import ChatOpenAI

from app.config import OPENAI_API_KEY


llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0,
    api_key=OPENAI_API_KEY
)


def generate_answer(question: str, context: str) -> str:
    prompt = f"""
You are a helpful knowledge assistant.

Answer the user's question using only the provided context.

Context:
{context}

Question:
{question}

If the answer is not available in the context, say:
"I don't have enough information in the provided documents."
"""

    response = llm.invoke(prompt)

    return response.content
