from langchain_chroma import Chroma
from app.rag.embeddings import get_embeddings


def get_retriever():
    vector_store = Chroma(
        persist_directory="data/chroma",
        embedding_function=get_embeddings()
    )

    return vector_store.as_retriever(
        search_kwargs={"k": 3}
    )
