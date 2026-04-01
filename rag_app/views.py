import os
import json

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from rag.pipeline import build_rag_pipeline

rag_chain = None
current_pdf_path = None


def index(request): # <WSGIRequest: GET '/'>
    # Request gồm các tham số
    # request.method
    # request.FILES
    # request.body
    # request.GET
    # request.POST
    return render(request, "index.html") # Tìm file index.html dựa trên settings biến nó thành HTTP Response 
                # trả về trình duyệt bên trong là HTML giống coi ở network -> Chrome nhận và render giao diện


@csrf_exempt # Bỏ qua kiểm tra csrf token 
def upload_pdf(request): # request <WSGIRequest: POST '/upload/'>
    global rag_chain
    global current_pdf_path
    if request.method == "POST":
        file = request.FILES.get("file") # lấy biến file PDF từ FormData từ hàm upload bên frontend.
        path = os.path.join("data", file.name)
        with open(path, "wb+") as f: # Tạo pdf mới trên ổ đĩa dưới dạng nhị phân
            for chunk in file.chunks(): # Ghi từng đoạn nhỏ vào file trong thư mục data dưới dạng binary
                f.write(chunk)
        qa_chain, vectorstore, chunks, documents = build_rag_pipeline(path)
        rag_chain = qa_chain # RetrieverQA
        current_pdf_path = path # Đường dẫn pdf
        # print(f"QA_CHAIN là cái này: {qa_chain}")
        # print(f"VECTORSTORE là cái này: {vectorstore}")
        # vec0 = vectorstore.index.reconstruct(0)
        # print("Vector thứ 0, 10 chiều đầu:", vec0[:10])
        # print("Số vector:", vectorstore.index.ntotal)
        # print("Chiều vector:", vectorstore.index.d)
        # print(f"CHUNKS là cái này: {chunks}")
        # print(f"DOCUMENTS là cái này: {documents}")
        return JsonResponse({ 
            "success": True,
            "pages": len(documents),
            "chunks": len(chunks),
            "pdf_path": path  
        })


@csrf_exempt
def ask_question(request): # request <WSGIRequest: POST '/ask/'>
     # urls gọi hàm này thì hàm này nhận request
    global rag_chain
    global current_pdf_path
    if request.method == "POST":
        if rag_chain is None: # Kiểm tra RetrieverQA
            return JsonResponse({
                "result": "Chưa upload PDF"
            })
        data = json.loads(request.body)
        query = data.get("query")
        result = rag_chain.invoke({ # Đặt câu hỏi và sinh câu trả lời và source_documents nếu bật
            "query": query
        })
        sources = [] # Nguồn tham khảo
        for doc in result["source_documents"]:
            sources.append({
                "page_content": doc.page_content,
                "metadata": doc.metadata
            })
        return JsonResponse({
            "result": result["result"],
            "pdf_path": current_pdf_path,
            "source_documents": sources
        })