from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from document_processing.pdf_loader import load_and_split_pdf
from document_processing.docx_loader import load_and_split_docx
from vector_store.faiss_store import create_faiss_vectorstore
from rag.retriever import get_retriever
from rag.llm import get_llm
from rag.promt import VIETNAMESE_PROMPT
from rag.hybrid_retriever import create_hybrid_retriever
import os


def build_rag_pipeline(file_path: str,chunk_size: int = 1000, chunk_overlap: int = 200, top_k: int = 3, fetch_k: int = 20, temperature: float = 0.7):
    file_format = os.path.splitext(file_path)[1].lower()
    if file_format == ".pdf":
        chunks, documents = load_and_split_pdf(file_path, chunk_size, chunk_overlap) # Load pdf và cắt chunk
    elif file_format == ".docx":
        chunks, documents = load_and_split_docx(file_path)
    vectorstore = create_faiss_vectorstore(chunks) # Tạo FAISS obj
    retriever = get_retriever(vectorstore, top_k, fetch_k) # Gán FAISS sang retriever
    hybrid_retriever = create_hybrid_retriever(
        vectorstore=vectorstore,
        chunks=chunks,
        top_k=top_k,
        fetch_k=fetch_k,
        bm25_weight=0.35,   
        vector_weight=0.65
    )
    llm = get_llm(temperature=temperature) # Tạo Ollama
    chatMemory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True, # Định dạng trả về true thì là Obj ([ HumanMessage("Hi"), AIMessage("Hello")]), false thì "Human: Hi\nAI: Hello"
        output_key="answer" # Lấy nội dung answer để lưu vào memory
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=hybrid_retriever,
        memory=chatMemory,
        chain_type="stuff", # Nối các chunk lại vảo promt
        combine_docs_chain_kwargs={"prompt": VIETNAMESE_PROMPT},
        return_source_documents=True,
        verbose=False  # debug memory
    )
    # qa_chain = RetrievalQA.from_chain_type( 
    #     llm=llm,
    #     retriever=retriever,
    #     chain_type="stuff", # Nối các chunk lại vảo promt
    #     chain_type_kwargs={"prompt": VIETNAMESE_PROMPT},  # Dùng prompt tiếng Việt
    #     return_source_documents=True,  # Hiển thị nguồn
    # )
    qa_chain.memory.clear() # reset memory mỗi lần 
    return (
        qa_chain,
        vectorstore,
        chunks,
        documents,
    )
