import os
import json
from datetime import datetime
import logging
import uuid
import time
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt
from rag.pipeline import build_rag_pipeline, query_rewriter, multi_hop_reasoning, self_rag_evaluate
from rag.retriever import get_retriever
from rag.llm import get_llm
from rag.hybrid_retriever import create_hybrid_retriever
from rag.corag import build_coRag


rag_chain = None
current_pdf_path = None
logger = None
# Thêm để xử lý mutil file
vectorstore_global = None   
top_k_global = 3           
fetch_k_global = 20         
all_chunks_global =[]
uploaded_file_name = []
# Thêm để xử lý self-RAG
llm = None
hybrid_retriever = None

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
    global rag_chain, current_pdf_path, logger
    global vectorstore_global, top_k_global, fetch_k_global
    global all_chunks_global, uploaded_file_name, llm, hybrid_retriever
    
    if request.method == "POST":
        chunk_size = int(request.POST.get("chunk_size", 1000))
        chunk_overlap = int(request.POST.get("chunk_overlap", 200))
        top_k = int(request.POST.get("top_k", 3))
        fetch_k = int(request.POST.get("fetch_k", 20))
        temperature = float(request.POST.get("temperature", 0.7))
        logger = create_logger()

        files = request.FILES.getlist("files") # lấy biến file PDF từ FormData từ hàm upload bên frontend.
        try:
            for file in files:
                logger.info(f"File: {file.name}")
                file_format = os.path.splitext(file.name)[1].lower()
                if file.size > 50 * 1024 * 1024:
                    logger.error("Lỗi File lớn hơn 50MB")
                    return JsonResponse({
                        "success": False,
                        "detail": f"Kích thước File {file.name} phải bé hơn 50MB"
                    }) 
                elif file_format not in [".pdf", ".docx"]:
                    logger.error("Lỗi sai định dạng File")
                    return JsonResponse({
                        "success": False,
                        "detail": "Chỉ hỗ trợ file PDF, DOCX! Vui lòng chọn file có định dạng .pdf, .docx"
                    }) 
                
            list_file_path = []
            for file in files:
                path = os.path.join("data", file.name)
                with open(path, "wb+") as f: # Tạo pdf mới trên ổ đĩa dưới dạng nhị phân
                    for chunk in file.chunks(): # Ghi từng đoạn nhỏ vào file trong thư mục data dưới dạng binary
                        f.write(chunk)
                list_file_path.append(path)

            qa_chain, hybrid_retriever, vectorstore, chunks, documents = build_rag_pipeline(list_file_path, chunk_size,chunk_overlap,top_k,fetch_k,temperature)
            rag_chain = qa_chain # RetrieverQA
            llm = get_llm(temperature=temperature) # Tạo Ollama
            vectorstore_global = vectorstore
            top_k_global = top_k              
            fetch_k_global = fetch_k
            all_chunks_global = chunks
            current_pdf_path = list_file_path  # Đường dẫn pdf
            uploaded_file_name = [os.path.basename(p) for p in list_file_path]
            hybrid_retriever =hybrid_retriever



            logger.info(f"Pages: {len(documents)}")
            logger.info(f"Chunks: {len(chunks)}")
            logger.info("UPLOAD THÀNH CÔNG")

            chat_sessions = get_or_create_chat_sessions(request)
            chat_id = chat_id = str(uuid.uuid4())
            file_title = files[0].name if len(files) == 1 else f"{len(files)} files"
            chat_sessions.append({
                "id": chat_id,
                "file": file_title,
                "files": uploaded_file_name,
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
                "current_chat_id":chat_id,
                "uploaded_files": uploaded_file_name,
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
    global rag_chain, current_pdf_path, logger
    global vectorstore_global, top_k_global, fetch_k_global
    global all_chunks_global, uploaded_file_name, llm, hybrid_retriever
    if request.method == "POST":
        if rag_chain is None: # Kiểm tra RetrieverQA
            if logger:
                logger.warning("User hỏi nhưng chưa upload PDF")
            return JsonResponse({
                "result": "Chưa upload PDF"
            })
        
        try: 
            data = json.loads(request.body)
            query = data.get("query") # Lấy câu hỏi
            selected_file_filter = data.get("file_name") # Lấy file filter user chọn
            filter_metadata = None
            if selected_file_filter: # ĐANG SAI HÀM
                filter_metadata = {"file_name": f"{selected_file_filter}"} # tên file_name phải match với bên pdf_loader
                retriever = create_hybrid_retriever( vectorstore_global, all_chunks_global, top_k_global, fetch_k_global, 
                                                        filter_metadata = filter_metadata) 
                rag_chain.retriever = retriever # Khi user đổi filter để hỏi thì sửa lại RetrieverQA
            logger.info(f"User: {query}")

            # Xử lý self-RAG
            rewritter_query = query_rewriter(llm, query) # Viết lại câu hỏi
            logger.info(f"Câu hỏi viết lại: {rewritter_query}")
            final_answer, all_document, confidence = multi_hop_reasoning( rag_chain, llm, rewritter_query) # Lấy final aw và all source

            # build_coRag(rag_chain, hybrid_retriever, llm, rewritter_query, True)
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
                        "content": final_answer or '',
                        "timestamp": datetime.now().strftime("%H:%M")
                    })
            request.session.modified = True # Báo session đã thay đổi

            sources = []
            # result["source_documents"]
            for i, doc in enumerate(all_document, 1):
                metadata = doc.metadata
                page = metadata.get("page") 
                # print(f"IDDD: {i} -- DOCCCCCCCC: {doc} -------- METADATAAAAA {doc.metadata} ----------- PAGEEEE: {page}")
                if isinstance(page, int):
                    page = page + 1  # vì nhiều loader bắt đầu từ 0

                sources.append({
                    "id": i,
                    "page_content": doc.page_content,           # 
                    "snippet": doc.page_content[:280] + "..." if len(doc.page_content) > 280 else doc.page_content,
                    "page": page,
                    "short_highlight": doc.page_content.replace('\n', ' ').strip(),
                    "source": os.path.basename(metadata.get("file_name", current_pdf_path)),
                    "source_path": "data/" + metadata.get("file_name", ""),
                    "chunk_index": metadata.get("chunk_index", "N/A"),
                    "full_text_for_highlight": doc.page_content.strip()
                })
            return JsonResponse({
                "result": final_answer,
                "confidence": confidence,
                "rewritten_query": rewritter_query,
                "pdf_path": current_pdf_path[0] if isinstance(current_pdf_path, list) else current_pdf_path,
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
    global rag_chain, current_pdf_path, logger
    global vectorstore_global, top_k_global, fetch_k_global
    global all_chunks_global, uploaded_file_name, llm

    if request.method == "POST":
        logger = create_logger() if logger is None else logger
        files = request.FILES.getlist("files") # lấy biến file PDF từ FormData từ hàm upload bên frontend.
        chunk_size = int(request.POST.get("chunk_size", 1000))
        chunk_overlap = int(request.POST.get("chunk_overlap", 200))
        top_k = int(request.POST.get("top_k", 3))
        fetch_k = int(request.POST.get("fetch_k", 20))
        temperature = float(request.POST.get("temperature", 0.7))
        
        try:
            list_file_path = [] # Danh sách các file có data/ sẽ truyền vào để build 

            if files: # Nếu thêm switch file mới chứ không phải sửa thông số
                for file in files:
                    logger.info(f"Đổi File: {file.name}")
                    file_format = os.path.splitext(file.name)[1].lower()
                    if file.size > 50 * 1024 * 1024:
                        logger.error("Lỗi File lớn hơn 50MB")
                        return JsonResponse({
                            "success": False,
                            "detail": f"Kích thước File {file.name} phải bé hơn 50MB"
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
                        list_file_path.append(path)

            else:   # Nếu chỉ sửa thông số
                for file_name in uploaded_file_name:
                    path = os.path.join("data", file_name)
                    list_file_path.append(path)

            qa_chain, hybrid_retriever, vectorstore, chunks, documents = build_rag_pipeline((list_file_path), chunk_size,chunk_overlap,top_k,fetch_k,temperature)
            rag_chain = qa_chain # RetrieverQA
            llm = get_llm()
            vectorstore_global = vectorstore
            top_k_global = top_k              
            fetch_k_global = fetch_k
            all_chunks_global = chunks
            current_pdf_path = list_file_path  # Đường dẫn pdf
            uploaded_file_name = [os.path.basename(p) for p in list_file_path]

            chat_sessions = get_or_create_chat_sessions(request)
            current_chat_id = request.session.get("current_chat_id")
            for chat in chat_sessions:
                if chat["id"] == current_chat_id:
                    chat["file"] = uploaded_file_name  # Đổi tên file trong session
                    break

            request.session.modified = True
            return JsonResponse({
                "success": True,
                "message": "Đã đổi sang tài liệu mới",
                "pages": len(documents),
                "chunks": len(chunks),
                "pdf_path": uploaded_file_name[0],
                "uploaded_file_name": uploaded_file_name,
                "current_chat_id": current_chat_id
            })

        except Exception as e:
            logger.error(f"Lỗi đổi tài liệu: {str(e)}")
            return JsonResponse({
                "success": False,
                "detail": "Lỗi khi xử lý tài liệu mới"
            })

    return JsonResponse({"error": "Method not allowed"}, status=405)