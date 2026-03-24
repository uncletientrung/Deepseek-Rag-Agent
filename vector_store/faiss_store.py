from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document
from rag.embedding import get_embedding_model

def create_faiss_vectorstore(chunks: List[Document]):
    """Tạo FAISS vector store từ list chunks."""
    embeddings = get_embedding_model()
    vectorstore = FAISS.from_documents(chunks, embeddings)
    return vectorstore

def save_faiss_vectorstore(vectorstore: FAISS, path: str = "vector_store/index"):
    """Lưu vector store xuống đĩa (tùy chọn sau này)."""
    vectorstore.save_local(path)

def load_faiss_vectorstore(path: str = "vector_store/index"):
    """Load từ đĩa (nếu bạn muốn persist)."""
    embeddings = get_embedding_model()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)