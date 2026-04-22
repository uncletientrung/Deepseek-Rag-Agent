from langchain.retrievers import EnsembleRetriever
from langchain_community.retrievers import BM25Retriever
from rag.retriever import get_retriever

def create_hybrid_retriever(vectorstore, chunks, top_k= 3, fetch_k=20, bm25_weight=0.35, vector_weight=0.65, filter_metadata =None):
    vector_retriever = get_retriever(vectorstore, k=top_k, fetch_k=fetch_k) # semantic search

    bm25_retriever = BM25Retriever.from_documents(chunks) # keyword search
    bm25_retriever.k = top_k

    combine_Search = EnsembleRetriever(
        retrievers=[bm25_retriever, vector_retriever],
        weights=[bm25_weight, vector_weight]
    )

    return combine_Search   