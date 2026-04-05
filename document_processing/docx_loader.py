from langchain_community.document_loaders import Docx2txtLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from typing import List
from langchain_core.documents import Document


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
    return chunks, documents