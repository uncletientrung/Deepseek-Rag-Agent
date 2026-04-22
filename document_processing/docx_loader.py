from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document
import os
from datetime import datetime


def load_and_split_docx(
    docx_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """Load DOCX và split thành chunks."""
    loader = Docx2txtLoader(docx_path)
    documents = loader.load() 
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    
    file_name = os.path.basename(docx_path)
    upload_time = datetime.now().isoformat()
    for i, chunk in enumerate(chunks):
        chunk.metadata.update({
            "file_name": file_name,              # tên file
            "file_type": "docx",               # loại file
            "chunk_index": i,                 # thứ tự chunk
            "upload_time": upload_time,       # thời gian upload
        })
    return chunks, documents