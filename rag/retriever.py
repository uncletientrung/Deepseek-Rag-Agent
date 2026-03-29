from langchain_community.vectorstores import FAISS
from typing import Optional


def get_retriever(vectorstore: FAISS, k: int = 3, fetch_k: int = 20):
    """Trả về retriever với search config."""
    return vectorstore.as_retriever(
        search_type="similarity", search_kwargs={"k": k, "fetch_k": fetch_k}
    )
