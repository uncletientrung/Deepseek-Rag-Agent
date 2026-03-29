from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import tempfile
import os
from rag.pipeline import build_rag_pipeline  # import hàm bạn đã có

app = FastAPI(title="RAG PDF API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Trong production nên giới hạn
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Lưu pipeline trong memory (cho 1 PDF)
rag_chain = None
current_pdf_path = None


@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    global rag_chain, current_pdf_path

    if not file.filename.endswith(".pdf"):
        raise HTTPException(400, detail="Chỉ hỗ trợ file PDF")

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(await file.read())
        pdf_path = tmp.name

    try:
        # Xây dựng pipeline
        qa_chain, vectorstore, chunks, documents = build_rag_pipeline(pdf_path)
        rag_chain = qa_chain
        current_pdf_path = pdf_path

        # Giả sử bạn có cách lấy số trang và chunks từ hàm load_and_split_pdf
        # Ở đây tạm hardcode, bạn có thể sửa để trả về thật
        return {
            "success": True,
            "message": "PDF processed successfully",
            "pdf_path": pdf_path,
            "pages": len(documents),  # ← bạn sửa thành giá trị thật
            "chunks": len(chunks),  # ← bạn sửa thành giá trị thật
        }
    except Exception as e:
        os.unlink(pdf_path)
        raise HTTPException(500, detail=str(e))


@app.post("/ask")
async def ask_question(data: dict):
    global rag_chain
    if not rag_chain:
        raise HTTPException(400, detail="Vui lòng upload PDF trước")

    query = data.get("query")
    if not query:
        raise HTTPException(400, detail="Thiếu query")

    result = rag_chain.invoke({"query": query})

    return {
        "result": result.get("result", "Không có câu trả lời"),
        "source_documents": [doc.dict() for doc in result.get("source_documents", [])],
    }


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="0.0.0.0", port=8000)
