from langchain_community.embeddings import HuggingFaceEmbeddings

def get_embedding_model():
    """Trả về embedding model (có thể cache)."""
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )