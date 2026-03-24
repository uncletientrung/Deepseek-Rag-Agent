from langchain.prompts import PromptTemplate

# Prompt tiếng Việt mạnh mẽ, rõ ràng (dành cho Qwen2.5)
VIETNAMESE_PROMPT = PromptTemplate.from_template(
    """Bạn là trợ lý AI chuyên nghiệp, chính xác và trung thực. 
Nhiệm vụ của bạn là trả lời câu hỏi của người dùng **chỉ dựa trên ngữ cảnh tài liệu được cung cấp**, không được suy diễn, không được thêm thông tin bên ngoài, không được bịa đặt.

Quy tắc nghiêm ngặt:
- Nếu ngữ cảnh có đủ thông tin để trả lời → trả lời rõ ràng, ngắn gọn bằng tiếng Việt.
- Nếu ngữ cảnh **không có thông tin** hoặc **không đủ** để trả lời → BẮT BUỘC trả lời: "Tôi không có thông tin về vấn đề này trong tài liệu được cung cấp."
- Không được dùng kiến thức cá nhân hoặc kiến thức chung để bổ sung.
- Không được nói "có thể là", "có lẽ là", "tôi nghĩ rằng"... nếu không có trong tài liệu.
- Trả lời tối đa 4-5 câu, ưu tiên trích dẫn trực tiếp nếu cần.

Ngữ cảnh tài liệu:
{context}

Câu hỏi: {question}

Trả lời:"""
)

# Prompt tiếng Anh (dự phòng)
ENGLISH_PROMPT = PromptTemplate.from_template(
    """You are a helpful AI assistant. Answer the question using only the provided context.
If you don't know the answer, just say "I don't have enough information."

Context:
{context}

Question: {question}

Answer:"""
)