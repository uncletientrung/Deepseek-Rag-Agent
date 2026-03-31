Project-LLMs-Rag-Agent/
│
├── manage.py
├── requirements.txt
├── README.md
├── .gitignore
│
├── rag_django/          ← project Django
│   ├── __init__.py
│   ├── settings.py
│   ├── urls.py
│   ├── asgi.py
│   └── wsgi.py
│
├── rag_app/             ← app Django
│   ├── templates/
│   │    └── index.html
│   │
│   ├── static/
│   │    └── styles/
│   │         └── style.css
│   │
│   ├── views.py
│   ├── urls.py
│   └── apps.py
│
├── rag/                 ← GIỮ NGUYÊN
│   ├── pipeline.py
│   ├── embedding.py
│   ├── retriever.py
│   ├── promt.py
│   └── llm.py
│
├── document_processing/ ← GIỮ NGUYÊN
│   └── pdf_loader.py
│
├── vector_store/        ← GIỮ NGUYÊN
│   └── faiss_store.py
│
├── data/
│
└── documentation/

============================ Luồng chạy: ========================
User mở web
    ↓
rag_django/urls.py          // Nơi chứa và định dạng các url GỐC của hệ thống
    ↓
rag_app/urls.py             // Nơi chứa các url nhánh của url gốc ""
    ↓
views.index                 // Gọi hàm index để render html 
    ↓
index.html                  // Trả về giao diện cho trình duyệt

Upload PDF                  // User upload -> chạy hàm js lấy được thông tin của file upload 
                            // Sau đó gửi url upload với thông tin của file upload
    ↓   
views.upload_pdf            // Tạo và Lưu pdf dưới dạng nhị phân vào thư mục data sau đó chạy RAG
    ↓
pipeline.py                      
    ↓
pdf_loader                  // Load PDF và cắt các chunk -> return thông tin chi tiết pdf có cả content,
                            // và thông tin chi tiết của các chunk có cả content sau cắt
    ↓
embedding                   // Cấu hình và trả về embedding model
    ↓
faiss                       
    ↓
llm
    ↓
rag_chain

Ask Question
    ↓
views.ask_question
    ↓
rag_chain.invoke
    ↓
FAISS search
    ↓
LLM
    ↓
Answer
    ↓
HTML

=======================================

🔷 Bước 1: Người dùng mở website

Người dùng vào:

http://127.0.0.1:8000
🔷 Bước 2: chạy rag_django/urls.py

File này là cổng chính của hệ thống.

from django.contrib import admin
from django.urls import path, include

urlpatterns = [
    path("", include("rag_app.urls")),
]
Ý nghĩa
127.0.0.1:8000
        ↓
rag_app.urls

➡️ Django chuyển sang app.

🔷 Bước 3: chạy rag_app/urls.py
from django.urls import path
from . import views

urlpatterns = [

    path("", views.index),

    path("upload/", views.upload_pdf),

    path("ask/", views.ask_question),
]
🎯 Ý nghĩa
URL	gọi function
/	index
/upload/	upload_pdf
/ask/	ask_question
🧭 Luồng sẽ là
URL → views.py
🔷 Bước 4: vào views.py

File này là bộ điều khiển logic.

🟢 Function 1
def index(request):
    return render(request, "index.html")
🎯 Ý nghĩa
User mở web
    ↓
urls.py
    ↓
index()
    ↓
render index.html
    ↓
Hiện giao diện
🧾 Trình duyệt nhận
index.html
style.css
script
🔷 Bước 5: Người dùng Upload PDF

trong HTML

fetch("/upload/", {
    method: "POST",
    body: formData
})
🎯 Trình duyệt gửi
POST /upload/
🔷 Bước 6: urls.py
path("upload/", views.upload_pdf)
🔷 Bước 7: vào views.py
def upload_pdf(request):
🧠 Luồng xử lý
Nhận file
↓
Lưu vào data/
↓
Build RAG pipeline
↓
Trả số trang
↓
Trả số chunks
↓
Trả pdf_path
📦 Code hoạt động
nhận file
file = request.FILES.get("file")
lưu file
path = os.path.join("data", file.name)
build pipeline
qa_chain, vectorstore, chunks, documents = build_rag_pipeline(path)
🔷 Bước 8: vào rag/pipeline.py

File này là trái tim hệ thống.

🧠 pipeline.py
load PDF
↓
split text
↓
embedding
↓
FAISS
↓
LLM
↓
RetrievalQA
🔷 Bước 9: pdf_loader.py
documents = load_pdf(path)
🎯
PDF
↓
Text
↓
Documents
🔷 Bước 10: split chunk
chunks = split_documents(documents)
🎯
Documents
↓
Chunks
🔷 Bước 11: embedding
embeddings = create_embedding()
🎯
Chunks
↓
Vector
🔷 Bước 12: FAISS
vectorstore = build_faiss(chunks, embeddings)
🎯
Vectors
↓
Database
🔷 Bước 13: LLM
llm = load_llm()
🎯
Qwen
🔷 Bước 14: RetrievalQA
qa_chain = RetrievalQA.from_chain_type(...)
🎯
Retriever
+
LLM
=
RAG
🔷 Bước 15: trả về views.py
return qa_chain
🔷 Bước 16: views lưu
rag_chain = qa_chain
🔷 Bước 17: trả kết quả về HTML
return JsonResponse({
    "success": True,
    "pages": len(documents),
    "chunks": len(chunks),
    "pdf_path": path
})
🔷 Bước 18: HTML nhận
data.pages
data.chunks
data.pdf_path
🎯 Hiển thị
PDF đã xử lý
10 pages
20 chunks
🔷 Bước 19: Người dùng hỏi
fetch("/ask/", {
    body: JSON.stringify({
        query: question
    })
})
🔷 Bước 20: urls.py
path("ask/", views.ask_question)
🔷 Bước 21: views.py
def ask_question(request):
🧠 Luồng
Nhận câu hỏi
↓
rag_chain.invoke()
↓
Retriever tìm chunks
↓
LLM trả lời
↓
Trả kết quả
🔷 Bước 22: RAG chạy
result = rag_chain.invoke({
    "query": query
})
🎯 bên trong
Question
↓
FAISS search
↓
Top chunks
↓
LLM
↓
Answer
🔷 Bước 23: trả về HTML
return JsonResponse({
    "result": result["result"]
})
🔷 Bước 24: HTML hiển thị
addMessage("ai", data.result)
🎯 Người dùng thấy
AI trả lời