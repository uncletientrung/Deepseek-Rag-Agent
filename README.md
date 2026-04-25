# Project: SmartDOC AI

Hệ thống hỏi đáp tài liệu sử dụng **RAG (Retrieval-Augmented Generation)** kết hợp nhiều kỹ thuật nâng cao như Hybrid Retrieval, Reranking, Multi-hop Reasoning và Self-RAG, CoRAG

---

## Tính năng chính

* Upload nhiều file PDF (Có hỗ trợ OCR)/ DOCX 
* Hybrid Search (BM25 + Vector Search)
* Multi-hop reasoning (chia nhỏ câu hỏi)
* Self-RAG evaluation (đánh giá độ tin cậy)
* Rerank bằng Cross Encoder
* Lưu lịch sử chat
* Switch tài liệu realtime

---

## Kiến trúc hệ thống

```
User → Django Views → RAG Pipeline
                        ↓
                Hybrid Retriever
              (BM25 + Vector Search)
                        ↓
                   Reranker
                        ↓
                      LLM
                        ↓
             Multi-hop + Self-RAG
                        ↓
                    Response
```

---

## Cấu trúc project

```
Project-LLMs-Rag-Agent/
│
├── manage.py
├── requirements.txt
│
├── rag_django/          # Django project
├── rag_app/             # Django app
│
├── rag/                 # Core RAG logic
├── document_processing/ # Load + OCR
├── vector_store/        # FAISS
│
├── data/                # File upload
├── Logging/             # Log hệ thống
```

---

## ⚙️ Yêu cầu hệ thống

* Python >= 3.10
* RAM >= 16GB
* pip
* virtualenv 

---

## 📦 Cài đặt

### 1. Clone project

```bash
git clone https://github.com/uncletientrung/Deepseek-Rag-Agent.git
cd Deepseek-Rag-Agent
```

### 2. Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt:

**Windows:**

```bash
venv\Scripts\activate
```

**Linux / Mac:**

```bash
source venv/bin/activate
```

### 3. Cài thư viện

```bash
pip install -r requirements.txt
```

### 4. Cấu hình LLM

Project dùng file `rag/llm.py`.

Ví dụ dùng Ollama:

```bash
ollama pull qwen2.5:3b
```

---

### 5. Chạy server

```bash
python manage.py migrate 

python manage.py runserver
```

---

### 6. Truy cập

```
http://127.0.0.1:8000/
```

---

## Cách sử dụng

### 1. Upload tài liệu

* Chọn file `.pdf` hoặc `.docx`
* Thiết lập:

  * chunk_size
  * chunk_overlap
  * top_k
  * fetch_k
  * temperature

---

### 2. Đặt câu hỏi

Hệ thống sẽ:

1. Rewrite câu hỏi
2. Retrieve tài liệu
3. Multi-hop reasoning
4. Self-evaluate
5. Trả kết quả + nguồn

---

### 3. Filter theo file

* Chọn file cụ thể để hỏi
* Hệ thống chỉ search trong file đó

---

### 4. Switch tài liệu

* Upload file mới hoặc đổi tham số
* Pipeline sẽ build lại

---

## Pipeline chi tiết

### Query Rewriting

* Tối ưu câu hỏi đầu vào

### Hybrid Retrieval

* FAISS + BM25

### Reranking

* Cross Encoder chọn kết quả tốt nhất

### Multi-hop Reasoning

* Tách câu hỏi nếu cần

### Self-RAG Evaluation

* Chấm điểm độ tin cậy (0 → 1)

---

## Logging

Log được lưu tại:

```
/Logging/
```

---

## Lưu ý

* File tối đa: 50MB
* Chỉ hỗ trợ PDF, DOCX

---
