from langchain.chains import RetrievalQA, ConversationalRetrievalChain
from langchain.memory import ConversationBufferMemory
from document_processing.pdf_loader import load_and_split_pdf
from document_processing.docx_loader import load_and_split_docx
from document_processing.ocr_loader import ocr_pdf
from document_processing.ocr_and_pdf_loader import ocr_and_pdf_loader
from vector_store.faiss_store import create_faiss_vectorstore
from rag.retriever import get_retriever
from rag.llm import get_llm
from rag.promt import VIETNAMESE_PROMPT
from rag.hybrid_retriever import create_hybrid_retriever
import os
import json
import re
from rag.reranker import CrossEncoderReranker
from rag.rerank_retriever import RerankRetriever

def query_rewriter(llm, query): # Viết lại câu hỏi
    prompt = f"""
        Bạn là hệ thống tối ưu truy vấn tìm kiếm tài liệu.

        Nhiệm vụ:
        Viết lại câu hỏi người dùng để phù hợp hơn cho việc tìm kiếm trong tài liệu.

        Quy tắc:
        - Trả lời bằng tiếng Việt
        - Không đổi nghĩa
        - Làm rõ ý hơn
        - Ngắn gọn
        - Chỉ trả về câu hỏi đã viết lại

        Câu hỏi: {query}

        Trả lời:
    """
    response = llm.invoke(prompt)
    return get_llm_text(response).strip()

def build_rag_pipeline(
        list_file_path, chunk_size=1000, chunk_overlap=200,
        top_k=3, fetch_k=15, temperature=0.7,
        filter_metadata=None # Dùng khi nếu user chọn filter cho multi file
    ):
    
    all_chunks = [] # Gộp hết các chunk trong các file thành 1 (cái này chứa các obj Document)
    all_documents = []
    for file_path in list_file_path:
        file_format = os.path.splitext(file_path)[1].lower()
        if file_format == ".pdf":
            chunks, documents = ocr_and_pdf_loader(file_path, chunk_size, chunk_overlap)  # Load pdf và cắt chunk
        elif file_format == ".docx":
            chunks, documents = load_and_split_docx(file_path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks) 
        all_documents.extend(documents)

    vectorstore = create_faiss_vectorstore(all_chunks) # Tạo FAISS obj
    # retriever = get_retriever(vectorstore, top_k, fetch_k, filter_metadata) # Gán FAISS sang retriever
    # hybrid_retriever = create_hybrid_retriever(   # Sử dụng bi-encoder
    #     vectorstore=vectorstore,
    #     chunks=all_chunks,
    #     top_k=top_k,
    #     fetch_k=fetch_k,
    #     bm25_weight=0.35,   
    #     vector_weight=0.65
    # )
    base_retriever = create_hybrid_retriever(
        vectorstore=vectorstore,    
        chunks=all_chunks,
        top_k=fetch_k, 
        fetch_k=fetch_k,
        bm25_weight=0.35,  vector_weight=0.65 
    )

    reranker = CrossEncoderReranker()
    hybrid_retriever = RerankRetriever(
        base_retriever=base_retriever,
        reranker=reranker,
        top_k=top_k,
        fetch_k=fetch_k
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



def multi_hop_reasoning(rag_chain, llm, written_query):
    result = rag_chain.invoke({"question": written_query})
    final_answer = result["answer"]
    all_source = result.get("source_documents", [])
    # Đánh giá lần 1
    confidence = self_rag_evaluate(llm, written_query, final_answer, all_source)
    print("---------- Đánh giá lần 1 ---------------")
    print(confidence)
    print(written_query)
    print(final_answer)

    if confidence > 0.7:
        return final_answer, all_source, confidence

    # Tách câu hỏi và đánh giá lần 2
    sub_queries = sub_query(llm, written_query)
    print("---------Danh sách câu hỏi sub -------------")
    print(sub_queries)
    collected_docs = []
    for q in sub_queries:
        print("---------- Câu hỏi sub ---------------")
        print(q)
        result = rag_chain.invoke({"question": q})
        collected_docs.extend(result.get("source_documents", []))

    collected_docs = documents_duyNhat(collected_docs) # Lấy docs duy nhất
    context = "\n\n".join([d.page_content for d in collected_docs]) # Tạo mảng ngữ cảnh với 2 cái space
    prompt = f"""
        Dựa trên các thông tin sau, hãy trả lời câu hỏi:

        Câu hỏi: {written_query}

        Tài liệu:
        {context}
        
        Quy tắc:
            - Chỉ sử dụng thông tin trong Context (Tài liệu)
            - Không suy diễn
            - Không thêm thông tin bên ngoài
            - Nếu không có thông tin thì trả lời:
            "Tôi không có thông tin về vấn đề này trong tài liệu được cung cấp."
                                                    
        Format:
            Giới thiệu ngắn (1 câu)
            Thông tin chính:
            - ...
            - ...
            - ...
                                                            
        Yêu cầu trình bày:
            - Trả lời bằng tiếng Việt
            - Ngắn gọn và dễ đọc khoảng 4-5 câu
            - Có câu mở đầu giới thiệu đối tượng
            - Sau đó liệt kê thông tin bằng gạch đầu dòng
            - Không liệt kê thô như PDF
            - Không được in ra các tiêu đề như:
            "Giới thiệu ngắn (1 câu)"
            "Thông tin chính"
            - Không được nhắc đến từ "nguồn", "cid"


        Trả lời:
    """

    response = llm.invoke(prompt)
    final_answer = get_llm_text(response)

    # Đánh giá lần 2
    confidence = self_rag_evaluate(llm, written_query, final_answer, collected_docs)
    print("---------- Đánh giá lần 2 ---------------")
    print(confidence)
    print(final_answer)

    return final_answer, collected_docs, confidence

def sub_query(llm, query):
    prompt = f"""
        Bạn là hệ thống phân rã câu hỏi cho multi-hop reasoning.

        Nhiệm vụ:
        Chia câu hỏi thành các câu hỏi nhỏ hơn để tìm kiếm từng bước.
        
        Quy tắc:
        - Câu hỏi là TIẾNG VIỆT
        - Tối đa 3-4 câu hỏi con
        - Không lặp lại câu gốc
        - Không đổi nghĩa gốc
        - Mỗi câu hỏi phải độc lập để truy vấn tài liệu
        - Bạn chỉ được trả về JSON list, KHÔNG markdown, KHÔNG giải thích.


        Ví dụ:
        Câu hỏi: "AI ảnh hưởng gì đến giáo dục và thị trường lao động?"
        Trả lời:
        ["AI ảnh hưởng gì đến giáo dục?", "AI ảnh hưởng gì đến thị trường lao động?"]

        Câu hỏi: {query}

        Trả lời:
    """
    response = llm.invoke(prompt)
    text = get_llm_text(response)
    text = re.sub(r"```json", "", text)
    text = re.sub(r"```", "", text).strip()
    match = re.search(r"\[.*\]", text, re.DOTALL)
    if match:
        text= match.group()
    print("----- Tách câu hỏi ở sub_q ------")
    print(text)
    print("-----------------")
    try:
        result = json.loads(match.group()) 
        if not result: # Nếu cắt trả về không có câu hỏi nào
            return [query]
        return result
    except Exception as e:
        print("Parse error:", e)
        return [query]

def self_rag_evaluate(llm, question, answer, contexts):
    context_text = "\n".join([doc.page_content for doc in contexts])

    prompt = f"""
        Bạn là hệ thống đánh giá câu trả lời (Self-RAG evaluator).

        Nhiệm vụ:
        Đánh giá mức độ đáng tin cậy của câu trả lời dựa trên context.

        Tiêu chí:
        - Đúng thông tin (correctness)
        - Có dựa trên context (groundedness)
        - Không bịa

        Trả về CHỈ 1 số từ 0.00 đến 1.00:
        - 0.00 = sai hoàn toàn
        - 1.00 = rất chính xác và đầy đủ

        Câu hỏi:
        {question}

        Context:
        {context_text}

        Câu trả lời:
        {answer}

        Điểm:
    """
    response = llm.invoke(prompt)
    text = get_llm_text(response)

    try:
        score = float(re.findall(r"\d+\.?\d*", text)[0])
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5
    
def get_llm_text(response): # Kiểm tra xem có hàm content không
    if hasattr(response, "content"):
        return response.content
    return response

def documents_duyNhat(docs): # Lấy docs với page_content duy nhất
    seen = set()
    unique_docs = []
    for doc in docs:
        key = doc.page_content.strip()         # dùng nội dung làm key để kiểm tra trùng
        if key not in seen:
            seen.add(key)
            unique_docs.append(doc)
    return unique_docs