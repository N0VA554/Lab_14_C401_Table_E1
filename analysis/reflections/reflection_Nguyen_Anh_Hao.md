# Individual Reflection Report — Analyst (Report) Role

| Thông tin | Chi tiết |
|---|---|
| **Họ và tên** | Nguyễn Anh Hào |
| **MSSV** | 2A202600131 |
| **Vai trò** | Analyst (Failure Analysis & QA) |
| **Ngày nộp** | 21/04/2026 |

---

## 1. Nhiệm vụ đảm nhận

Với vai trò **Analyst**, tôi chịu trách nhiệm chính trong việc "soi lỗi" và tổng hợp kết quả để đưa ra cái nhìn khách quan về chất lượng của Agent. Các nhiệm vụ cụ thể bao gồm:
- Thực hiện phân tích thất bại chuyên sâu (Failure Analysis) cho toàn bộ các case bị điểm thấp.
- Áp dụng phương pháp công nghiệp minh bạch (**5 Whys**) để tìm ra nguyên nhân gốc rễ (Root Cause) của các vấn đề trong hệ thống RAG.
- Chạy công cụ kiểm soát chất lượng `check_lab.py` để đảm bảo bài nộp đáp ứng đầy đủ các tiêu chuẩn kỹ thuật và định dạng.
- Tổng hợp báo cáo `failure_analysis.md`, đề xuất các hành động cải tiến (Action Plan) dựa trên dữ liệu thực tế từ benchmark.

---

## 2. Những đóng góp và kết quả đạt được

### 2.1 Xây dựng Báo cáo Phân tích Thất bại chuyên sâu

Thay vì chỉ liệt kê các case lỗi, tôi đã tiến hành phân nhóm lỗi theo các danh mục kỹ thuật để đội ngũ phát triển dễ dàng ưu tiên xử lý:
- **Phân loại lỗi:** Xác định tỉ lệ lỗi do Retrieval (chiếm đa số) so với lỗi do Generation hoặc Hallucination.
- **Phân tích theo mức độ khó:** Chỉ ra rằng hệ thống hiện đang gặp khó khăn nhất ở nhóm câu hỏi **Medium** (yêu cầu kết hợp thông tin) và **Very Hard** (yêu cầu suy luận nhiều bước).

### 2.2 Áp dụng phương pháp 5 Whys

Tôi đã trực tiếp mổ xẻ 3 trường hợp tệ nhất (điểm 1/5) để tìm ra các "nút cổ chai" kỹ thuật:
1. **Lỗi Paraphrasing gap:** Phân tích lý do tại sao keyword retrieval thất bại khi câu hỏi và tài liệu không khớp từ khóa.
2. **Lỗi Metadata-aware retrieval:** Chỉ ra tầm quan trọng của việc đánh chỉ mục số điều luật trong các tài liệu pháp lý/quy chế.
3. **Lỗi Parent-child mapping:** Đề xuất việc giữ lại cấu ngữ cảnh rộng hơn thay vì chỉ lấy các mảnh (chunks) nhỏ rời rạc.

### 2.3 Đảm bảo chất lượng bài lab (QA)

Tôi sử dụng script `check_lab.py` để thực hiện các bước kiểm tra cuối cùng:
- Kiểm tra tính đầy đủ của các file artifact bắt buộc (`summary.json`, `benchmark_results.json`).
- Xác định sự hiện diện của các metrics quan trọng như Hit Rate và Agreement Rate.
- Đảm bảo tính nhất quán giữa dữ liệu benchmark và các nhận định trong báo cáo.

---

## 3. Bài học kinh nghiệm và Kỹ năng học được

1. **Tầm quan trọng của dữ liệu sạch:** Một hệ thống RAG tốt bắt đầu từ việc chunking và gắn metadata thông minh. Nếu bước "Retrieval" sai, dù LLM có mạnh đến đâu cũng sẽ trả về kết quả kém hoặc ảo tưởng.
2. **Tư duy phản biện (Critical Thinking):** Việc thực hiện 5 Whys giúp tôi thoát khỏi việc đổ lỗi cho "AI yếu" mà chuyển sang tìm lỗi ở "Cấu trúc dữ liệu" hoặc "Chiến lược truy xuất".
3. **Quy trình QA tự động:** Việc sử dụng các script kiểm tra định dạng giúp tiết kiệm thời gian và tránh các sai sót thủ công không đáng có trước khi bàn giao sản phẩm.
4. **Tối ưu hóa chi phí:** Hiểu được cách cân bằng giữa chất lượng (Multi-judge) và chi phí (Two-stage filtering) là một kỹ năng quan trọng khi triển khai AI trong thực tế.

---

## 4. Kết luận

Vai trò Analyst giúp tôi hiểu rõ rằng trong một hệ thống AI phức tạp, việc nắm bắt được **tại sao nó sai** cũng quan trọng như việc làm cho nó chạy đúng. Những phân tích trong báo cáo `failure_analysis.md` sẽ là tiền đề quan trọng để đội ngũ chuyển đổi sang sử dụng Vector Embedding và giải quyết triệt để các vấn đề về ngữ nghĩa trong tương lai.
