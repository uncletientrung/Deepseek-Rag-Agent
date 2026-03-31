from langchain_community.vectorstores import FAISS
from typing import List
from langchain_core.documents import Document
from rag.embedding import get_embedding_model


def create_faiss_vectorstore(chunks: List[Document]):
    """Tạo FAISS vector store từ list chunks."""
    embeddings = get_embedding_model() # Trả về embedding model
    vectorstore = FAISS.from_documents(chunks, embeddings) # Chuyển hóa chunk thành vector db
    return vectorstore # Trả về FAISS obj chứa index (tất cả vector) và document mapping vector (cái để cái text gốc hay là page_content)

def save_faiss_vectorstore(vectorstore: FAISS, path: str = "vector_store/index"): # Lưu FAISS vào ổ đĩa chưa dùng
    vectorstore.save_local(path)

def load_faiss_vectorstore(path: str = "vector_store/index"):#Load FAISS từ ổ đĩa
    embeddings = get_embedding_model()
    return FAISS.load_local(path, embeddings, allow_dangerous_deserialization=True)
