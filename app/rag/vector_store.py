from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma

from app.config import settings


def create_vector_store(documents):
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY
    )

    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory="data/chroma"
    )

    return vector_store


def load_vector_store():
    embeddings = OpenAIEmbeddings(
        api_key=settings.OPENAI_API_KEY
    )

    return Chroma(
        persist_directory="data/chroma",
        embedding_function=embeddings
    )
