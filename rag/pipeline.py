from langchain.chains import RetrievalQA
from document_processing.pdf_loader import load_and_split_pdf
from vector_store.faiss_store import create_faiss_vectorstore
from rag.retriever import get_retriever
from rag.llm import get_llm
from rag.promt import VIETNAMESE_PROMPT


def build_rag_pipeline(pdf_path: str):
    chunks, documents = load_and_split_pdf(pdf_path) # Load pdf và cắt chunk
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
