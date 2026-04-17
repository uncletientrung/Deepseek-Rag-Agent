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
        chunk_size = int(request.POST.get("chunk_size", 1000))
        chunk_overlap = int(request.POST.get("chunk_overlap", 200))
        top_k = int(request.POST.get("top_k", 3))
        fetch_k = int(request.POST.get("fetch_k", 20))
        temperature = float(request.POST.get("temperature", 0.7))
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
            qa_chain, vectorstore, chunks, documents = build_rag_pipeline(path, chunk_size,chunk_overlap,top_k,fetch_k,temperature)
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
            request.session.modified = True # Báo cho django biết session đã bị thay đổi và lưu lại

            return JsonResponse({ 
                "success": True,
                "pages": len(documents),
                "chunks": len(chunks),
                "pdf_path": path,
                "current_chat_id":chat_id
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
            logger.info(f"Bot: Trả lời thành công")
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

@csrf_exempt
def delete_chat(request): # Xóa chat dựa trên id
    if request.method == "POST":
        try:
            data = json.loads(request.body)
            chat_id = data.get("remove_chat_id") # Lấy id chat cần xóa
            current_chat_id = request.session.get("current_chat_id") 
            if not chat_id:
                return JsonResponse({"success": False, "message": "Thiếu chat_id"}, status=400)

            chat_sessions = get_or_create_chat_sessions(request)
            updated_sessions = [] # Lọc bỏ chat cần xóa
            for chat in chat_sessions:
                if chat["id"] != chat_id:
                    updated_sessions.append(chat)
            
            request.session['chat_sessions'] = updated_sessions # Update lại lịch sử chat
            if request.session.get('current_chat_id') == chat_id: # nếu xóa chat đang nhắn
                current_chat_id = -1
                    
            request.session.modified = True
            return JsonResponse({
                "success": True,
                "message": "Đã xóa cuộc trò chuyện",
                "chat_sessions": updated_sessions,
                "current_chat_id": current_chat_id
                
            })
        except Exception as e:
            return JsonResponse({
                "success": False,
                "message": "Có lỗi khi xóa cuộc trò chuyện"
            }, status=500)
    
    return JsonResponse({"error": "Lỗi ở xóa chat trong Views"}, status=405)

@csrf_exempt
def switch_document(request):
    global rag_chain
    global current_pdf_path
    global logger

    if request.method == "POST":
        logger = create_logger() if logger is None else logger
        path = request.POST.get("path")
        print(path)
        chunk_size = int(request.POST.get("chunk_size", 1000))
        chunk_overlap = int(request.POST.get("chunk_overlap", 200))
        top_k = int(request.POST.get("top_k", 3))
        fetch_k = int(request.POST.get("fetch_k", 20))
        temperature = float(request.POST.get("temperature", 0.7))
        try:
            if path:
                file_name = os.path.basename(path)
                print(file_name)
            else:
                file = request.FILES.get("file")
                if not file:
                    return JsonResponse({"success": False, "detail": "Không tìm thấy file"}, status=400)

                file_format = os.path.splitext(file.name)[1].lower()
                
                if file.size > 50 * 1024 * 1024:
                    return JsonResponse({"success": False, "detail": "File phải nhỏ hơn 50MB"})
                if file_format not in [".pdf", ".docx"]:
                    return JsonResponse({"success": False, "detail": "Chỉ hỗ trợ PDF và DOCX"})

                path = os.path.join("data", file.name)
                with open(path, "wb+") as f:
                    for chunk in file.chunks():
                        f.write(chunk)
                file_name = file.name
            
            qa_chain, vectorstore, chunks, documents = build_rag_pipeline(path, chunk_size,chunk_overlap,top_k,fetch_k,temperature)
            rag_chain = qa_chain
            current_pdf_path = path

            logger.info(f"Đổi tài liệu thành công: {file_name} | Pages: {len(documents)} | Chunks: {len(chunks)}")
            chat_sessions = get_or_create_chat_sessions(request)
            current_chat_id = request.session.get("current_chat_id")
            for chat in chat_sessions:
                if chat["id"] == current_chat_id:
                    chat["file"] = file_name  # Đổi tên file trong session
                    break

            request.session.modified = True
            return JsonResponse({
                "success": True,
                "message": "Đã đổi sang tài liệu mới",
                "file_name": file_name,
                "pages": len(documents),
                "chunks": len(chunks),
                "pdf_path": path,
                "current_chat_id": current_chat_id
            })

        except Exception as e:
            logger.error(f"Lỗi đổi tài liệu: {str(e)}")
            return JsonResponse({
                "success": False,
                "detail": "Lỗi khi xử lý tài liệu mới"
            })

    return JsonResponse({"error": "Method not allowed"}, status=405)