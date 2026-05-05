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


retriever_all_file = None
per_file_retrievers = {} # Mỗi file 1 retriever

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
        top_k=3, fetch_k=10, temperature=0.7,
        filter_metadata=None # Dùng khi nếu user chọn filter cho multi file
    ):
    global retriever_all_file, per_file_retrievers

    all_chunks = [] # Gộp hết các chunk trong các file thành 1 (cái này chứa các obj Document)
    all_documents = []
    for file_path in list_file_path:
        file_format = os.path.splitext(file_path)[1].lower()
        if file_format == ".pdf":
            chunks, documents = ocr_and_pdf_loader(file_path, chunk_size, chunk_overlap)
        elif file_format == ".docx":
            chunks, documents = load_and_split_docx(file_path, chunk_size, chunk_overlap)
        all_chunks.extend(chunks) 
        all_documents.extend(documents)

    vectorstore = create_faiss_vectorstore(all_chunks) # Sử dụng embedding để chuyển các chunks thành vector db rồi lưu vào FAISS

    hybrid_retriever_reranker = create_hybrid_retriever( # Rerank trong này
        vectorstore=vectorstore,    
        chunks=all_chunks,
        top_k=top_k, 
        fetch_k=fetch_k,
        bm25_weight=0.35,  vector_weight=0.65,
        filter_metadata= None
    )
    retriever_all_file = hybrid_retriever_reranker # Retriever cho tất cả file
    if len(list_file_path) >= 2:
        for file_path in list_file_path:   # hoặc extract từ all_chunks
            file_name = os.path.basename(file_path)
            file_chunks = [doc for doc in all_chunks if doc.metadata.get("file_name") == file_name]
            per_file_retrievers[file_name] = create_hybrid_retriever(
                vectorstore=vectorstore,
                chunks=file_chunks,           # chỉ truyền chunks của file đó
                top_k=top_k,
                fetch_k=fetch_k,
                bm25_weight=0.4,  vector_weight=0.6,
                filter_metadata={"file_name": file_name}
            )

    llm = get_llm(temperature=temperature) # Tạo Ollama
    chatMemory = ConversationBufferMemory(
        memory_key="chat_history",
        return_messages=True, # Định dạng trả về true thì là Obj ([ HumanMessage("Hi"), AIMessage("Hello")]), false thì "Human: Hi\nAI: Hello"
        output_key="answer" # Lấy nội dung answer để lưu vào memory
    )

    qa_chain = ConversationalRetrievalChain.from_llm(
        llm=llm,
        retriever=retriever_all_file,
        memory=chatMemory,
        chain_type="stuff", # Nối các chunk lại vảo promt
        combine_docs_chain_kwargs={"prompt": VIETNAMESE_PROMPT},
        return_source_documents=True,
        verbose=False  # debug memory
    )

    qa_chain.memory.clear() # reset memory mỗi lần 

    return (
        qa_chain,
        retriever_all_file,
        vectorstore,
        all_chunks,
        all_documents,
    )


def build_multi_hop_pipeline(rag_chain, llm, written_query, selected_file_filter=None):
    global retriever_all_file, per_file_retrievers
    if selected_file_filter and selected_file_filter.lower() != None:
        current_retriever = per_file_retrievers.get(selected_file_filter)
    else:
        current_retriever = retriever_all_file
    if not current_retriever:
        current_retriever = retriever_all_file

    rag_chain.retriever = current_retriever

    result = rag_chain.invoke({"question": written_query})
    chat_history = rag_chain.memory.chat_memory.messages
    final_answer = result["answer"]
    all_source = result.get("source_documents", [])
    if len(final_answer) < 30 and "không" in final_answer.lower():
        print("Trả lời quá ngắn")   
        print(final_answer)
        return multi_hop_reasoning(rag_chain, llm, written_query, chat_history)
    if "không có thông tin" in final_answer.lower():
        print("Không có câu trả lời")
        print(final_answer)
        return multi_hop_reasoning(rag_chain, llm, written_query, chat_history)
    
    # # Đánh giá lần 1
    confidence = self_rag_evaluate(llm, written_query, final_answer, all_source)
    print("---------- Đánh giá lần 1 ---------------")
    print(confidence)
    print(written_query)
    print(final_answer)
    if confidence < 0.5:
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
        - Tối đa 2 câu hỏi phụ.
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
    if not contexts or not answer or not answer.strip():
        return 0.0
    
    context_text = "\n\n".join([f"--- Document {i+1} ---\n{doc.page_content}" 
                               for i, doc in enumerate(contexts)])
    
    prompt = f"""Bạn là một evaluator chuyên đánh giá Hallucination và độ bám sát câu hỏi trong hệ thống RAG.

Nhiệm vụ: Đánh giá câu trả lời dựa trên **CHỈ 2 tiêu chí** sau:

1. **Hallucination (Không bịa đặt)**: 
   - Câu trả lời có chứa thông tin nào KHÔNG có trong Context không?
   - Mọi thông tin, sự kiện, số liệu trong câu trả lời phải được hỗ trợ trực tiếp bởi Context.

2. **Relevance & Faithfulness (Độ bám sát câu hỏi)**:
   - Câu trả lời có trực tiếp trả lời câu hỏi không?
   - Có đi lạc đề hoặc trả lời thừa/thiếu so với ý câu hỏi không?

**Quy tắc chấm điểm nghiêm ngặt (0.0 - 1.0)**:
- 0.0 - 0.3: Có hallucination rõ ràng hoặc hoàn toàn không trả lời đúng câu hỏi
- 0.4 - 0.6: Có một phần hallucination hoặc chỉ trả lời được một phần câu hỏi
- 0.7 - 0.85: Không hallucination, trả lời khá tốt nhưng còn thiếu sót nhỏ
- 0.9 - 1.0: Không hallucination, trả lời chính xác, đầy đủ và bám sát câu hỏi

Hãy suy nghĩ từng bước:
1. Liệt kê các claim chính trong câu trả lời.
2. Kiểm tra từng claim có được hỗ trợ bởi Context không.
3. Đánh giá mức độ trả lời đúng câu hỏi.
4. Đưa ra điểm số cuối cùng.

Câu hỏi:
{question}

Context:
{context_text}

Câu trả lời cần đánh giá:
{answer}

Trả về **CHỈ** một dòng theo đúng format sau, không thêm gì khác:

Điểm: 0.XX
Giải thích: (tối đa 1-2 câu ngắn gọn)
"""

    try:
        response = llm.invoke(prompt)
        text = get_llm_text(response)

        # Tìm điểm số
        score_match = re.search(r"Điểm:\s*([0-9]*\.?[0-9]+)", text)
        if score_match:
            score = float(score_match.group(1))
            return min(max(score, 0.0), 1.0)
        
        # Fallback: tìm bất kỳ số thập phân nào hợp lý
        scores = re.findall(r"0\.\d+|1\.0|1", text)
        if scores:
            return min(max(float(scores[0]), 0.0), 1.0)

        return 0.5

    except Exception as e:
        print(f"Lỗi self_rag_evaluate: {e}")
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