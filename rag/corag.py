import os
import json
import re


def get_llm_text(response): # Kiểm tra xem có hàm content không
    if hasattr(response, "content"):
        return response.content
    return response
# =========================
# 1. REWRITE (sub-queries)
# =========================
def rewrite_query(llm, query):
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

# =========================
# 2. PLAN (lập kế hoạch truy xuất)
# =========================
def plan_retrieval(llm, query, sub_queries):
    prompt = f"""
        Bạn là một hệ thống lập kế hoạch cho CoRAG.

        Câu hỏi người dùng:
        {query}

        Các truy vấn con:
        {sub_queries}

        Hãy tạo một kế hoạch truy xuất theo từng bước để tìm thông tin.

        Yêu cầu:
        - Chia thành các bước rõ ràng
        - Mỗi bước nên mô tả cần tìm gì
        - Trả về dạng danh sách các bước
    """

    response = llm.invoke(prompt)
    return get_llm_text(response).strip()

# =========================
# 3. MULTI-HOP RETRIEVAL
# =========================
def multi_hop_retrieval(hybrid_retriever, sub_queries):
    all_docs = []
    for q in sub_queries:
        docs = hybrid_retriever.get_relevant_documents(q)
        all_docs.append(docs)

    return all_docs


# =========================
# 4. GỘP NGỮ CẢNH
# =========================
def merge_docs(all_docs):
    seen = set()
    context = []
    for docs in all_docs:
        for d in docs:
            text = d.page_content.strip()
            if text not in seen:
                seen.add(text)
                context.append(text)

    return "\n\n".join(context)


# =========================
# 5. SUY LUẬN + TẠO CÂU TRẢ LỜI
# =========================
def generate_answer(llm, query, context):
    prompt = f"""
        Bạn là một trợ lý AI thông minh.

        Dựa vào ngữ cảnh bên dưới, hãy trả lời câu hỏi của người dùng.

        Ngữ cảnh:
        {context}

        Câu hỏi:
        {query}

        Yêu cầu:
        - Trả lời rõ ràng, có cấu trúc
        - Nếu thông tin không đủ, hãy nói "không đủ dữ liệu để trả lời chính xác"
        """
    response = llm.invoke(prompt)
    return get_llm_text(response).strip()


# =========================
# 6. BEST-OF-N (tuỳ chọn)
# =========================
def best_of_n(llm, query, context, n=3):
    candidates = []

    for i in range(n):
        prompt = f"""
Hãy trả lời câu hỏi theo một cách khác (phiên bản {i+1}).

Ngữ cảnh:
{context}

Câu hỏi:
{query}

Yêu cầu:
- Trả lời tự nhiên
- Mỗi câu trả lời nên có cách diễn đạt khác nhau
"""
        response = llm.invoke(prompt)
        ans = get_llm_text(response).strip()
        candidates.append(ans)

    judge_prompt = f"""
Bạn là một hệ thống đánh giá câu trả lời.

Câu hỏi:
{query}

Dưới đây là các câu trả lời:

{json.dumps(candidates, ensure_ascii=False, indent=2)}

Hãy chọn câu trả lời tốt nhất.

Chỉ trả về câu trả lời được chọn, không giải thích.
"""
    response = llm.invoke(judge_prompt)
    return get_llm_text(response)


# =========================
# MAIN CO-RAG PIPELINE
# =========================
def build_coRag(rag_chain, hybrid_retriever, llm, rewritter_query, use_best_of_n=False):

    print("--------------------Bắt Đầu chạy CORAG ------------------")
    # 1. Rewrite câu hỏi
    print("--------------------Tách câu hỏi ------------------")
    sub_queries = rewrite_query(llm, rewritter_query)
    print(sub_queries)

    # 2. Lập kế hoạch truy xuất
    print("--------------------Lập kế hoạch ------------------") # PLAN KHÔNG CẦN THIẾT TRONG CORAG
    plan = plan_retrieval(llm, rewritter_query, sub_queries)
    print(plan)

    # 3. Multi-hop retrieval
    print("--------------------Multi docs ------------------")
    all_docs = multi_hop_retrieval(hybrid_retriever, sub_queries)
    print(all_docs)

    # 4. Gộp ngữ cảnh
    print("--------------------Gộp ngữ cảnh ------------------")
    context = merge_docs(all_docs)
    print(context)

    # 5. Sinh câu trả lời
    if use_best_of_n:
        answer = best_of_n(llm, rewritter_query, context)
    else:
        answer = generate_answer(llm, rewritter_query, context)

    print("--------------------Trả lời ------------------")
    print(answer)
    return {
        "query": rewritter_query,
        "sub_queries": sub_queries,
        "plan": plan,
        "context": context,
        "answer": answer
    }