from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os
from datetime import datetime

def load_and_split_pdf(
    pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """Load PDF và split thành chunks + metadata."""
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()   # Load PDF

    text_splitter = RecursiveCharacterTextSplitter( # Cắt chunk
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)

    file_name = os.path.basename(pdf_path)
    upload_time = datetime.now().isoformat()
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "file_name": file_name,              # tên file
            "file_type": "pdf",               # loại file
            "chunk_index": i,                 # thứ tự chunk
            "upload_time": upload_time,       # thời gian upload
        })

    return chunks, documents
