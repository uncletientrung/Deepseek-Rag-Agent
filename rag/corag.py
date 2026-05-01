import json
import re
from typing import List, Dict, Any, Tuple

def get_llm_text(response):
    if hasattr(response, "content"):
        return response.content
    return response


def rewrite_initial_query(llm, query: str) -> List[str]: # Tách câu hỏi
    prompt = f"""
        Bạn là hệ thống phân rã câu hỏi cho multi-hop reasoning.
        Hãy chia câu hỏi sau thành tối đa 3 câu hỏi con độc lập, rõ ràng bằng tiếng Việt.
        Câu hỏi gốc: {query}
        Yêu cầu:
        - Chỉ trả về JSON list, không giải thích, không markdown.
        - Mỗi câu hỏi phải có thể truy vấn độc lập.

        Ví dụ: 
        ["Câu hỏi con 1?", "Câu hỏi con 2?", "Câu hỏi con 3?"]

        Trả lời:
    """
    response = llm.invoke(prompt)
    text = get_llm_text(response)
    text = re.sub(r"```json|```", "", text).strip()
    try:
        match = re.search(r"\[.*\]", text, re.DOTALL)
        if match:
            return json.loads(match.group())
    except:
        pass
    return [query]


def should_stop(llm, original_query: str, context: str, reasoning_trace: List[Dict]) -> bool:
    """Kiểm tra xem đã đủ thông tin để trả lời chưa"""
    if len(reasoning_trace) >= 5:  # giới hạn tối đa
        return True
    
    prompt = f"""
        Câu hỏi gốc: {original_query}

        Ngữ cảnh đã thu thập được:
        {context[-1500:]}  # chỉ lấy phần mới nhất để tránh token quá dài

        Các bước suy luận trước đó:
        {json.dumps(reasoning_trace[-3:], ensure_ascii=False, indent=2)}

        Hỏi: Đã đủ thông tin để trả lời chính xác câu hỏi gốc chưa?
        Trả lời chỉ bằng "YES" hoặc "NO".
    """
    resp = get_llm_text(llm.invoke(prompt)).strip().upper()
    return "YES" in resp


def iterative_corag(llm, hybrid_retriever, query: str, max_hops: int = 5, use_best_of_n: bool = False) -> Dict:
    print("-------------------- BẮT ĐẦU ITERATIVE CoRAG ------------------")
    all_docs = []
    context_parts = [] # Chỉ chứa page_content của docs để đưa vào llm
    reasoning_trace = [] # Ghi lại quá trình suy luận
    unique_docs = set() # page_content của docs duy nhất

    # Bước 1: Phân rã ban đầu
    sub_queries = rewrite_initial_query(llm, query)
    current_query = sub_queries[0] if sub_queries else query

    for hop in range(max_hops): # Tạo vòng lặp cho multi hop
        print(f"\nHop {hop+1}/{max_hops} - Query: {current_query}")
        
        docs = hybrid_retriever.get_relevant_documents(current_query) # Retrieve dựa trên câu hỏi hiện tại
        new_docs = []
        for doc in docs:
            text = doc.page_content.strip()
            if text not in unique_docs and text:
                unique_docs.add(text)
                new_docs.append(doc)
                context_parts.append(text)
        all_docs.extend(new_docs)
        
        # Intermediate reasoning
        current_context = "\n\n".join(context_parts[-4:])  # chỉ 4 phần tử cuối
        reason_prompt = f"""
            Dựa trên thông tin vừa thu thập được, hãy suy luận xem nó góp phần trả lời câu hỏi gốc như thế nào.

            Câu hỏi gốc: {query}
            Thông tin mới: {current_context}

            Suy luận ngắn gọn (2-4 câu):
        """
        reasoning = get_llm_text(llm.invoke(reason_prompt)).strip()
        reasoning_trace.append({
            "hop": hop + 1,
            "sub_query": current_query,
            "reasoning": reasoning,
            "docs_count": len(new_docs)
        })
        
        print(f"Reasoning: {reasoning[:200]}...")

        # Kiểm tra đã đủ thông tin chưa nếu rồi thì dừng
        full_context = "\n\n".join(context_parts)
        if should_stop(llm, query, full_context, reasoning_trace):
            print("   → Đủ thông tin, dừng sớm.")
            break

        # Reformulate query cho bước tiếp theo
        reform_prompt = f"""
            Câu hỏi gốc: {query}

            Ngữ cảnh hiện tại: {full_context[-1200:]}

            Các suy luận trước: {json.dumps([r["reasoning"] for r in reasoning_trace], ensure_ascii=False, indent=2)}

            Hãy sinh ra **một câu hỏi tiếp theo** (bằng tiếng Việt) cần tìm kiếm để thu thập thêm thông tin quan trọng nhất cho câu trả lời.

            Chỉ trả về đúng một câu hỏi, không giải thích.
        """
        next_query = get_llm_text(llm.invoke(reform_prompt)).strip()
        current_query = next_query if next_query else query

    # ================== TẠO CÂU TRẢ LỜI CUỐI CÙNG ==================
    final_context = "\n\n".join(context_parts)


    answer_prompt = f"""
        Bạn là trợ lý trả lời chính xác dựa trên tài liệu.

        Ngữ cảnh:
        {final_context}

        Câu hỏi: {query}

        Hãy trả lời rõ ràng, có cấu trúc, trích dẫn thông tin chính xác từ ngữ cảnh.
        Nếu không đủ dữ liệu, hãy nói rõ.
    """
    answer = get_llm_text(llm.invoke(answer_prompt)).strip()

    # ================== TRẢ VỀ NGUỒN THAM KHẢO ==================
    sources = []
    for i, doc in enumerate(all_docs[:15], 1):   # giới hạn số nguồn
        meta = doc.metadata
        sources.append({
            "id": i,
            "page_content": doc.page_content,
            "snippet": doc.page_content[:280] + "..." if len(doc.page_content) > 280 else doc.page_content,
            "page": meta.get("page", "N/A"),
            "source": meta.get("file_name", meta.get("source", "unknown")),
            "chunk_index": meta.get("chunk_index", "N/A")
        })

    print("-------------------- CoRAG HOÀN THÀNH ------------------")
    
    return {
        "answer": answer,
        "sources": sources,
        "trace": reasoning_trace,      # để debug hoặc hiển thị quá trình suy nghĩ
        "context": final_context[:8000] # nếu cần
    }
