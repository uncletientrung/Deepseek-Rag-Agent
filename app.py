import streamlit as st
import tempfile
import os
from rag.pipeline import build_rag_pipeline

st.set_page_config(page_title="RAG System", layout="wide")
st.title("📚 RAG System with LLMs")
st.write("Upload PDF và hỏi câu hỏi về nội dung tài liệu")

# Upload file
uploaded_file = st.file_uploader("Upload PDF", type="pdf")

if uploaded_file is not None:
    # Lưu tạm
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
        tmp_file.write(uploaded_file.read())
        pdf_path = tmp_file.name

    st.success("PDF uploaded successfully!")

    # Build pipeline chỉ 1 lần (dùng session_state)
    if "qa_chain" not in st.session_state:
        with st.spinner("Đang xử lý PDF, tạo embeddings và xây dựng RAG pipeline..."):
            qa_chain, vectorstore = build_rag_pipeline(pdf_path)
            st.session_state.qa_chain = qa_chain
            st.session_state.vectorstore = vectorstore  # nếu muốn dùng sau

        st.success("RAG system ready!")

    # Hỏi đáp
        # Hỏi đáp
    st.subheader("Hỏi câu hỏi")
    question = st.text_input("Nhập câu hỏi của bạn:")

    if st.button("Hỏi"):
        if question:
            with st.spinner("Đang suy nghĩ..."):
                # Dùng invoke thay vì run
                result = st.session_state.qa_chain.invoke({"query": question})
            
            st.subheader("📝 Câu trả lời:")
            st.write(result["result"])

            # Hiển thị nguồn tài liệu
            with st.expander("📚 Nguồn tài liệu tham khảo"):
                for i, doc in enumerate(result.get("source_documents", []), 1):
                    st.markdown(f"**Nguồn {i}:**")
                    content = doc.page_content.strip()
                    st.write(content[:700] + "..." if len(content) > 700 else content)
                    st.write("---")
        else:
            st.warning("Vui lòng nhập câu hỏi")

# Cleanup temp file khi session kết thúc (tùy chọn)
if "pdf_path" in locals() and os.path.exists(pdf_path):
    os.unlink(pdf_path)