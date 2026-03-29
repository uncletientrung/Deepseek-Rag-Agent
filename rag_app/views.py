import os
import json

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from rag.pipeline import build_rag_pipeline

rag_chain = None
current_pdf_path = None


def index(request):
    # Request gồm các tham số
    # request.method
    # request.FILES
    # request.body
    # request.GET
    # request.POST
    return render(request, "index.html") # Tìm file index.html dựa trên settings biến nó thành HTTP Response 
                # trả về trình duyệt bên trong là HTML giống coi ở network -> Chrome nhận và render giao diện


@csrf_exempt
def upload_pdf(request):
    global rag_chain
    global current_pdf_path
    if request.method == "POST":
        file = request.FILES.get("file") # lấy file PDF từ frontend.
        path = os.path.join("data", file.name)
        with open(path, "wb+") as f:
            for chunk in file.chunks():
                f.write(chunk)

        qa_chain, vectorstore, chunks, documents = build_rag_pipeline(path)
        rag_chain = qa_chain
        current_pdf_path = path

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