import streamlit as st

from langchain_community.document_loaders import PDFPlumberLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS

from langchain_community.llms import Ollama
from langchain.chains import RetrievalQA

import os
import tempfile

# ===============================
# Streamlit UI
# ===============================

st.set_page_config(page_title="RAG System", layout="wide")

st.title("📚 RAG System with LLMs")
st.write("Upload PDF và hỏi câu hỏi về nội dung tài liệu")

# Upload PDF
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

# ===============================
# Nếu có file
# ===============================

if uploaded_file is not None:

    # Lưu file tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # ===============================
    # Load PDF
    # ===============================

    loader = PDFPlumberLoader(pdf_path)
    documents = loader.load()

    st.write(f"Loaded {len(documents)} pages")

    # ===============================
    # Split Text
    # ===============================

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )

    chunks = text_splitter.split_documents(documents)

    st.write(f"Created {len(chunks)} chunks")

    # ===============================
    # Embedding
    # ===============================

    embedder = HuggingFaceEmbeddings(
        model_name="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

    st.write("Embedding model loaded")

    # ===============================
    # FAISS
    # ===============================

    vector = FAISS.from_documents(chunks, embedder)

    st.success("FAISS Vector Database created")

    # ===============================
    # Retriever
    # ===============================

    retriever = vector.as_retriever(
        search_type="similarity",
        search_kwargs={
            "k": 3,
            "fetch_k": 20
        }
    )

    st.success("Retriever ready")

    # ===============================
    # LLM
    # ===============================

    llm = Ollama(
        model="qwen2.5:3b",
        temperature=0.7,
        top_p=0.9,
        repeat_penalty=1.1
    )

    st.success("Ollama Qwen2.5 loaded")

    # ===============================
    # QA Chain
    # ===============================

    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        chain_type="stuff"
    )

    st.success("RAG system ready")

    # ===============================
    # Ask Question
    # ===============================

    st.subheader("Ask Question")

    question = st.text_input("Enter your question")

    if st.button("Ask"):

        if question:

            with st.spinner("Thinking..."):

                answer = qa_chain.run(question)

            st.subheader("Answer")

            st.write(answer)

        else:
            st.warning("Please enter a question")