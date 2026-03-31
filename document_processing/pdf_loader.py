from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


def load_and_split_pdf(
    pdf_path: str, chunk_size: int = 1000, chunk_overlap: int = 200
) -> List[Document]:
    """Load PDF và split thành chunks."""
    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()   # Load PDF

    text_splitter = RecursiveCharacterTextSplitter( # Cắt chunk
        chunk_size=chunk_size, chunk_overlap=chunk_overlap
    )
    chunks = text_splitter.split_documents(documents)
    return chunks, documents
