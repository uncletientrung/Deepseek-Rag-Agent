from langchain.chains import RetrievalQA
from document_processing.pdf_loader import load_and_split_pdf
from vector_store.faiss_store import create_faiss_vectorstore
from rag.retriever import get_retriever
from rag.llm import get_llm
from rag.promt import VIETNAMESE_PROMPT

def build_rag_pipeline(pdf_path: str):
    """Xây dựng toàn bộ pipeline từ PDF path."""
    # 1. Load & Split
    chunks = load_and_split_pdf(pdf_path)

    # 2. Create Vector Store
    vectorstore = create_faiss_vectorstore(chunks)

    # 3. Retriever
    retriever = get_retriever(vectorstore)

    # 4. LLM
    llm = get_llm()

    # 5. Tạo QA Chain với Prompt tiếng Việt
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff",
        chain_type_kwargs={"prompt": VIETNAMESE_PROMPT},   # Dùng prompt tiếng Việt
        return_source_documents=False                       # Bật để hiển thị nguồn
    )

    return qa_chain, vectorstore  # trả về cả chain và vectorstore nếu cần