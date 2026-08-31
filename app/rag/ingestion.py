from pathlib import Path

from langchain_community.document_loaders import TextLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter


def load_documents(directory: str = "data/documents"):
    documents = []

    for file_path in Path(directory).glob("*.txt"):
        loader = TextLoader(str(file_path), encoding="utf-8")
        documents.extend(loader.load())

    return documents


def split_documents(documents):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=800,
        chunk_overlap=120
    )

    return splitter.split_documents(documents)
