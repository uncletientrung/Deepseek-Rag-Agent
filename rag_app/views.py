import os
import json
from datetime import datetime
import logging
import uuid

from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rag.pipeline import build_rag_pipeline

rag_chain = None
current_pdf_path = None
logger = None

def get_or_create_chat_sessions(request):
    if 'chat_sessions' not in request.session:
        request.session['chat_sessions'] = []
    return request.session['chat_sessions']

def create_logger():
    log_folder = "Logging"
    if not os.path.exists(log_folder):
        os.makedirs(log_folder)
    log_file_name = datetime.now().strftime("%Y-%m-%d_%H-%M-%S") + ".txt"
    log_file_path = os.path.join(log_folder, log_file_name)
    logger = logging.getLogger("rag_logger") # Tạo 1 logger tên là rag
    logger.setLevel(logging.INFO) # Nếu không set thì nó chỉ ghi từ level warning trở lên mà info < warning
    if logger.hasHandlers():    # Kiểm tra đã có handler chưa (handler quyệt định logger ghi vào đâu)
        logger.handlers.clear()  # xóa handler cũ
    file_handler = logging.FileHandler(log_file_path, encoding="utf-8")
    formatter = logging.Formatter(
        "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    )
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger

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
    global logger
    
    if request.method == "POST":
        logger = create_logger()
        file = request.FILES.get("file") # lấy biến file PDF từ FormData từ hàm upload bên frontend.
        file_format = os.path.splitext(file.name)[1].lower()
        request.session['file_name']= file.name
        try:
            logger.info(f"File: {file.name}")
            if file.size > 50 * 1024 * 1024:
                logger.error("Lỗi File lớn hơn 50MB")
                return JsonResponse({
                    "success": False,
                    "detail": "Kích thước File phải bé hơn 50MB"
                }) 
            elif file_format not in [".pdf", ".docx"]:
                logger.error("Lỗi sai định dạng File")
                return JsonResponse({
                    "success": False,
                    "detail": "Chỉ hỗ trợ file PDF, DOCX! Vui lòng chọn file có định dạng .pdf, .docx"
                }) 
            
            path = os.path.join("data", file.name)
            with open(path, "wb+") as f: # Tạo pdf mới trên ổ đĩa dưới dạng nhị phân
                for chunk in file.chunks(): # Ghi từng đoạn nhỏ vào file trong thư mục data dưới dạng binary
                    f.write(chunk)
            qa_chain, vectorstore, chunks, documents = build_rag_pipeline(path)
            rag_chain = qa_chain # RetrieverQA
            current_pdf_path = path # Đường dẫn pdf

            logger.info(f"Pages: {len(documents)}")
            logger.info(f"Chunks: {len(chunks)}")
            logger.info("UPLOAD THÀNH CÔNG")

            chat_sessions = get_or_create_chat_sessions(request)
            chat_id = chat_id = str(uuid.uuid4())
            chat_sessions.append({
                "id": chat_id,
                "file": file.name,
                "created_at": datetime.now().strftime("%H:%M"),
                "history": []
            })

            request.session['current_chat_id'] = chat_id
            request.session.modified = True

            return JsonResponse({ 
                "success": True,
                "pages": len(documents),
                "chunks": len(chunks),
                "pdf_path": path  
            })
        except Exception as e:
            logger.error(f"Lỗi upload: {str(e)}")
            return JsonResponse({
                "success": False,
                "detail": "Lỗi không thể kết nối LLM"
            })


@csrf_exempt
def ask_question(request): # request <WSGIRequest: POST '/ask/'>
     # urls gọi hàm này thì hàm này nhận request
    global rag_chain
    global current_pdf_path
    global logger
    if request.method == "POST":
        if rag_chain is None: # Kiểm tra RetrieverQA
            if logger:
                logger.warning("User hỏi nhưng chưa upload PDF")
            return JsonResponse({
                "result": "Chưa upload PDF"
            })
        
        try: 
            data = json.loads(request.body)
            query = data.get("query")
            logger.info(f"User: {query}")
            result = rag_chain.invoke({ # Đặt câu hỏi và sinh câu trả lời và source_documents nếu bật
                "query": query
            })
            logger.info(f"Bot: {result['result']}")
            logger.info("------------------------------------------------------------")

            # Lưu vào session chat
            chat_sessions = get_or_create_chat_sessions(request)
            current_chat_id = request.session.get("current_chat_id")
            for chat in chat_sessions:
                if chat["id"] == current_chat_id:
                    chat["history"].append({
                        "role": "user",
                        "content": query,
                        "timestamp": datetime.now().strftime("%H:%M")
                    })

                    chat["history"].append({
                        "role": "ai",
                        "content": result['result'],
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
            request.session.modified = True # Báo session đã thay đổi

            sources = [] # Nguồn tham khảo
            for doc in result["source_documents"]:
                sources.append({
                    "page_content": doc.page_content,
                    "metadata": doc.metadata
                })
            return JsonResponse({
                "result": result["result"],
                "pdf_path": current_pdf_path,
                "source_documents": sources,
                "chat_sessions": chat_sessions,
                "current_chat_id": request.session.get("current_chat_id")
            })
        except Exception as e:
            logger.error(f"Lỗi khi hỏi: {str(e)}")
            return JsonResponse({
                "result": "Có lỗi xảy ra"
            })

@csrf_exempt
def get_chat_history(request):
    if request.method == "GET":
        chat_sessions = get_or_create_chat_sessions(request)
        return JsonResponse({
            "chat_sessions": chat_sessions,
            "current_chat_id": request.session.get("current_chat_id")
        })
    return JsonResponse({"error": "Lỗi lấy chat history ở Views"}, status=405)