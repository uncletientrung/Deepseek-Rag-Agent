import os
import json
from datetime import datetime

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from rag.pipeline import build_rag_pipeline

rag_chain = None
current_pdf_path = None
log_file_path = None


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
        try:
            write_log("\n===== Upload PDF =====")
            write_log(f"Time: {datetime.now()}")
            write_log(f"File: {file.name}")
            if file.size > 50 * 1024 * 1024:
                write_log("Lỗi: Kích thước File lớn hơn 50MB")
                return JsonResponse({
                    "success": False,
                    "detail": "Kích thước File phải bé hơn 50MB"
                }) 
            path = os.path.join("data", file.name)
            with open(path, "wb+") as f: # Tạo pdf mới trên ổ đĩa dưới dạng nhị phân
                for chunk in file.chunks(): # Ghi từng đoạn nhỏ vào file trong thư mục data dưới dạng binary
                    f.write(chunk)
            qa_chain, vectorstore, chunks, documents = build_rag_pipeline(path)
            rag_chain = qa_chain # RetrieverQA
            current_pdf_path = path # Đường dẫn pdf

            write_log(f"Pages: {len(documents)}")
            write_log(f"Chunks: {len(chunks)}")
            write_log("Upload thành công")

            return JsonResponse({ 
                "success": True,
                "pages": len(documents),
                "chunks": len(chunks),
                "pdf_path": path  
            })
        except:
            write_log(f"Lỗi upload: {str(e)}")

            return JsonResponse({
                "success": False,
                "detail": "Lỗi không thể kết nối LLM"
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

def write_log(content):
    global log_file_path
    if log_file_path is None:
        log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
        log_file_path = os.path.join("Logging", log_file_name)
        with open(log_file_path, "w", encoding="utf-8") as f:
            f.write(content + "\n")
    else:
        with open(log_file_path, "a", encoding="utf-8") as f:
            f.write(content + "\n")