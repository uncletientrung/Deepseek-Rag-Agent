from langchain_community.embeddings import HuggingFaceEmbeddings


def get_embedding_model():
    """Trả về embedding model"""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2", # Tải model
        model_kwargs={"device": "cpu"}, # Cấu hình CPU
        encode_kwargs={"normalize_embeddings": True}, # Chuẩn hóa vector về 1 để chính xác
    )
