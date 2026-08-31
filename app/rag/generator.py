from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI


def generate_answer(question: str, context: str) -> str:
    prompt = ChatPromptTemplate.from_template(
        """
You are a helpful enterprise knowledge assistant.

Answer the user's question using ONLY the provided context.
If the answer cannot be found in the context, say:
"I could not find this information in the provided documents."

Context:
{context}

Question:
{question}

Answer:
"""
    )

    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    chain = prompt | llm

    response = chain.invoke(
        {
            "context": context,
            "question": question,
        }
    )

    return response.content
