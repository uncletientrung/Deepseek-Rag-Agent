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


def build_rag_pipeline(
        list_file_path, chunk_size=1000, chunk_overlap=200,
        top_k=3, fetch_k=20, temperature=0.7,
        filter_metadata=None # Dùng khi nếu user chọn filter cho multi file
    ):
    
    all_chunks = [] # Gộp hết các chunk trong các file thành 1 (cái này chứa các obj Document)
    all_documents = []
    for file_path in list_file_path:
        file_format = os.path.splitext(file_path)[1].lower()
        if file_format == ".pdf":
            chunks, documents = load_and_split_pdf(file_path, chunk_size, chunk_overlap)  # Load pdf và cắt chunk
        elif file_format == ".docx":
            chunks, documents = load_and_split_docx(file_path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks) 
        all_documents.extend(documents)

    vectorstore = create_faiss_vectorstore(all_chunks) # Tạo FAISS obj
    # retriever = get_retriever(vectorstore, top_k, fetch_k, filter_metadata) # Gán FAISS sang retriever
    hybrid_retriever = create_hybrid_retriever(
        vectorstore=vectorstore,
        chunks=all_chunks,
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
        all_chunks,
        all_documents,
    )
