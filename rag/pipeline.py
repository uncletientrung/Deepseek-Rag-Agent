from langchain.chains import RetrievalQA
from document_processing.pdf_loader import load_and_split_pdf
from document_processing.docx_loader import load_and_split_docx
from vector_store.faiss_store import create_faiss_vectorstore
from rag.retriever import get_retriever
from rag.llm import get_llm
from rag.promt import VIETNAMESE_PROMPT
import os


def build_rag_pipeline(file_path: str):
    file_format = os.path.splitext(file_path)[1].lower()
    if file_format == ".pdf":
        chunks, documents = load_and_split_pdf(file_path) # Load pdf và cắt chunk
    elif file_format == ".docx":
        chunks, documents = load_and_split_docx(file_path)
    vectorstore = create_faiss_vectorstore(chunks) # Tạo FAISS obj
    retriever = get_retriever(vectorstore) # Gán FAISS sang retriever
    llm = get_llm() # Tạo Ollama

    qa_chain = RetrievalQA.from_chain_type( 
        llm=llm,
        retriever=retriever,
        chain_type="stuff", # Nối các chunk lại vảo promt
        chain_type_kwargs={"prompt": VIETNAMESE_PROMPT},  # Dùng prompt tiếng Việt
        return_source_documents=True,  # Hiển thị nguồn
    )

    return (
        qa_chain,
        vectorstore,
        chunks,
        documents,
    )
