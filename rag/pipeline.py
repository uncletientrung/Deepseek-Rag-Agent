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
        hybrid_retriever,
        vectorstore,
        all_chunks,
        all_documents,
    )


def build_multi_hop_pipeline(rag_chain, llm, written_query):
    result = rag_chain.invoke({"question": written_query})
    chat_history = rag_chain.memory.chat_memory.messages
    final_answer = result["answer"]
    all_source = result.get("source_documents", [])
    if len(final_answer) < 30 and "không" in final_answer.lower():
        print("Trả lời quá ngắn")
        return multi_hop_reasoning(rag_chain, llm, written_query, chat_history)
    if "không có thông tin" in final_answer.lower():
        print("Không có câu trả lời")
        return multi_hop_reasoning(rag_chain, llm, written_query, chat_history)
    
    # Đánh giá lần 1
    confidence = self_rag_evaluate(llm, written_query, final_answer, all_source)
    print("---------- Đánh giá lần 1 ---------------")
    print(confidence)
    print(written_query)
    print(final_answer)
    if confidence < 0.7:
        return multi_hop_reasoning(rag_chain, llm, written_query, chat_history)
    
    return final_answer, all_source, confidence


def multi_hop_reasoning(rag_chain, llm, written_query, chat_history):
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
    prompt = VIETNAMESE_PROMPT.format(
        context=context,
        question=written_query,
        chat_history =chat_history,
    )
    response = llm.invoke(prompt)
    final_answer = get_llm_text(response)

    # Đánh giá lần 2
    confidence = self_rag_evaluate(llm, written_query, final_answer, collected_docs)
    print("---------- Đánh giá lần 2 ---------------")
    print(confidence)
    print(final_answer)

    return final_answer, collected_docs, confidence

def sub_query(llm, query):
    prompt = f"""Bạn là chuyên gia phân rã câu hỏi (Query Decomposition) cho multi-hop RAG.
        Nhiệm vụ: Phân tích câu hỏi tiếng Việt và chia thành các câu hỏi phụ (sub-questions) cần thiết để trả lời toàn diện và logic.

        ### Quy tắc nghiêm ngặt:
        - Câu hỏi gốc là tiếng Việt → Tất cả sub-questions cũng phải bằng tiếng Việt.
        - Tối đa 3 câu hỏi phụ (ưu tiên 2-3 nếu cần thiết).
        - Mỗi sub-question phải **độc lập**, có thể dùng để tìm kiếm tài liệu riêng lẻ.
        - Không được thay đổi ý nghĩa gốc của câu hỏi.
        - Không được thêm thông tin mới hoặc suy diễn ngoài câu hỏi gốc.
        - Không lặp lại câu hỏi gốc.
        - Nếu câu hỏi chỉ cần 1 bước → trả về list chứa đúng 1 câu hỏi (gần giống gốc nhưng rõ ràng hơn).
        - Phải giữ nguyên thông tin quan trọng (tên riêng, sự kiện, khái niệm...).
        - Trả về **CHỈ** một JSON array hợp lệ, không có bất kỳ chữ nào khác.

        ### Ví dụ 1:
        Câu hỏi: "AI ảnh hưởng gì đến giáo dục và thị trường lao động?"
        Trả lời:
        ["AI ảnh hưởng như thế nào đến giáo dục?", "AI ảnh hưởng như thế nào đến thị trường lao động?"]

        ### Ví dụ 2:
        Câu hỏi: "Ai là người thắng cuộc bầu cử tổng thống Mỹ năm 2024?"
        Trả lời:
        ["Ai thắng cuộc bầu cử tổng thống Mỹ năm 2024?"]

        ### Ví dụ 3:
        Câu hỏi: "Quy trình sản xuất và xuất khẩu cà phê của Việt Nam hiện nay ra sao?"
        Trả lời:
        ["Quy trình sản xuất cà phê tại Việt Nam hiện nay như thế nào?", "Việt Nam xuất khẩu cà phê sang những thị trường nào và tình hình ra sao?"]

        Câu hỏi cần phân rã:
        {query}

        Trả về đúng định dạng JSON sau (không thêm bất kỳ nội dung nào khác):

        [
        "câu hỏi phụ 1",
        "câu hỏi phụ 2",
        ...
        ]
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
    context_text = "\n\n".join([f"--- Document {i+1} ---\n{doc.page_content}" 
                               for i, doc in enumerate(contexts)])
    
    prompt = f"""Bạn là một Self-RAG Evaluator nghiêm ngặt và khách quan bậc cao.
Nhiệm vụ của bạn là đánh giá chất lượng câu trả lời dựa trên context được cung cấp.

Tiêu chí đánh giá (đánh giá theo thang điểm chi tiết):

1. **Groundedness / Faithfulness (Không bịa đặt)**: 
   - Câu trả lời có hoàn toàn dựa trên thông tin có trong Context không?
   - Mọi claim/fact trong câu trả lời đều phải được hỗ trợ trực tiếp bởi Context.
   - Không được thêm thông tin từ kiến thức chung hoặc suy luận quá mức.

2. **Correctness (Tính chính xác)**:
   - Câu trả lời có đúng và chính xác với thông tin trong Context không?
   - Có mâu thuẫn với Context không?

3. **Completeness (Tính đầy đủ)**:
   - Câu trả lời có bao quát đầy đủ các thông tin quan trọng cần thiết để trả lời câu hỏi không?
   - Có bỏ sót thông tin quan trọng có trong Context không?

4. **Relevance (Tính liên quan)**:
   - Câu trả lời có trực tiếp giải quyết câu hỏi không?

**Quy tắc nghiêm ngặt**:
- Nếu câu trả lời chứa **bất kỳ thông tin nào không có trong Context** → điểm giảm mạnh (hallucination).
- Nếu câu trả lời mâu thuẫn với Context → điểm rất thấp.
- Nếu câu trả lời chỉ đúng một phần → điểm trung bình.
- Chỉ đạt điểm cao khi **toàn bộ** nội dung đều được hỗ trợ bởi Context và trả lời đầy đủ + chính xác.

Hãy suy nghĩ từng bước một (Chain-of-Thought):

Bước 1: Phân tích các claim chính trong câu trả lời.
Bước 2: Kiểm tra từng claim có được hỗ trợ bởi Context không (trích dẫn cụ thể nếu có).
Bước 3: Đánh giá tổng thể theo 4 tiêu chí trên.
Bước 4: Đưa ra điểm số cuối cùng.

Câu hỏi:
{question}

Context:
{context_text}

Câu trả lời cần đánh giá:
{answer}

Bây giờ, suy nghĩ kỹ và trả về **CHỈ** một số thập phân từ 0.00 đến 1.00 theo format sau:

Điểm: X.XX

Giải thích ngắn gọn (tùy chọn, tối đa 2-3 câu):
"""

    response = llm.invoke(prompt)
    text = get_llm_text(response)
    
    try:
        # Tìm số thập phân (hỗ trợ cả 0.0 và 1.00, 0.85...)
        scores = re.findall(r"Điểm:\s*(\d+\.?\d*)|(\d+\.?\d*)", text)
        for s in scores:
            score_str = next((x for x in s if x), None)
            if score_str:
                score = float(score_str)
                return min(max(score, 0.0), 1.0)
        # Fallback: tìm bất kỳ số nào hợp lệ
        score = float(re.findall(r"\d+\.?\d*", text)[-1])
        return min(max(score, 0.0), 1.0)
    except:
        return 0.5  # fallback an toàn
    
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