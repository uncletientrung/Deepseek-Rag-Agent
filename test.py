from rank_bm25 import BM25Okapi

# =========================
# 1. Dữ liệu (documents)
# =========================
docs = [
    "AI là lĩnh vực nghiên cứu máy thông minh",
    "Machine learning là một phần của AI",
    "Python được dùng trong AI và data science",
    "Hôm nay trời đẹp không liên quan AI"
]

# =========================
# 2. Tokenize (tách từ)
# =========================
tokenized_docs = [doc.lower().split() for doc in docs]
print(tokenized_docs)

# =========================
# 3. Tạo BM25 model
# =========================
bm25 = BM25Okapi(tokenized_docs)

# =========================
# 4. Query
# =========================
query = "AI machine learning"
tokenized_query = query.lower().split()

# =========================
# 5. Tính score
# =========================
scores = bm25.get_scores(tokenized_query)

# =========================
# 6. In kết quả
# =========================
for i, score in enumerate(scores):
    print(f"Doc {i}: score = {score:.4f} | {docs[i]}")

# =========================
# 7. Lấy top 2
# =========================
top_n = bm25.get_top_n(tokenized_query, docs, n=2)

print("\nTop results:")
for doc in top_n:
    print("-", doc)