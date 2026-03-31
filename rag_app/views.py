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
        rag_chain = qa_chain
        current_pdf_path = path
        # print(f"QA_CHAIN là cái này: {qa_chain}")
        # print(f"VECTORSTORE là cái này: {vectorstore}")
        # print(f"CHUNKS là cái này: {chunks}")
        # print(f"DOCUMENTS là cái này: {documents}")
        return JsonResponse({ 
            "success": True,
            "pages": len(documents),
            "chunks": len(chunks),
            "pdf_path": path  
        })


@csrf_exempt
def ask_question(request):

    global rag_chain
    global current_pdf_path

    if request.method == "POST":

        if rag_chain is None:
            return JsonResponse({
                "result": "Chưa upload PDF"
            })

        data = json.loads(request.body)

        query = data.get("query")

        result = rag_chain.invoke({
            "query": query
        })

        return JsonResponse({
            "result": result["result"],
            "pdf_path": current_pdf_path
        })