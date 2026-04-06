User mở web
    ↓
loadHistory()               // Chạy hàm window.addEventListener('load') để lấy lịch sử từ session của BE
    ↓
views.get_chat_history()    // Trả về chat_session (Lịch sử chat) và current_chat_id (Id chat hiện tại)
   ↓
renderHistoryItem()         // Sau khi có được các lịch sử chat rồi thì sẽ render ra các item lịch sử
    ↓
loadConversation(chat.id)   // Cài cho mỗi item lịch sử 1 action khi bấm vòa sẽ chạy load lịch sử

Upload PDF                  //  User upload -> Tạo ra mảng chat_sessions mới or lấy cái chat_sessions cũ
                            // đã xử lý từ lần chạy trước và session current_chat_id để giữ id hiện tại
                            // Cấu trúc 1 chat_session có dạng:
                                chat_sessions = [
                                    {
                                        id: 69d3209b-d5f0-8399-8cd3-c2ca33fa7d87,
                                        file: "pdf1.pdf",
                                        history: [user, ai, user, ai],
                                        created_at: "10:20"
                                    },
                                    {
                                        id: 69d3209b-3123-8399-8cd3-c2ca33fa7d87,
                                        file: "pdf2.pdf",
                                        history: [user, ai],
                                        created_at: "10:25"
                                    }
                                ]

Ask Question                // Người dùng nhập câu hỏi
                            // Lấy chat_session và current_chat_id duyệt qua từng từng id kiểm tra nếu
                            // đùng là id đó thì thêm lịch sử chat vào history
                            // Trả về chat_session và current_chat_id để FE hiển thị lịch sử nếu được chọn




