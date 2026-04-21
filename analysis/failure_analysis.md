# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark và Hiệu suất Hệ thống

Báo cáo này cung cấp cái nhìn chi tiết về hiệu suất của hệ thống RAG (Retrieval-Augmented Generation) thông qua việc so sánh giữa hai phiên bản: V1 (Base) và V2 (Optimized). Quá trình benchmark được thực hiện trên tập dữ liệu gồm 50 câu hỏi thử nghiệm với các độ khó khác nhau.

| Chỉ số Hiệu suất | Phiên bản V1 (Base) | Phiên bản V2 (Optimized) | Thay đổi |
|:---|:---:|:---:|:---:|
| **Tổng số lượng test cases** | 50 | 50 | - |
| **Số lượng Pass / Fail** | 31 / 19 | 36 / 14 | +5 Pass |
| **Tỷ lệ vượt qua (Pass Rate)** | 62.0% | 72.0% | +10.0% |
| **Điểm Judge trung bình (1-5)** | 3.250 / 5.0 | 3.560 / 5.0 | +0.31 |
| **Tỷ lệ tìm chính xác (Hit Rate)** | 62.0% | 62.0% | 0.0% |
| **Chỉ số MRR** | 0.453 | 0.468 | +0.015 |
| **Độ trễ trung bình (Latency)** | 1.71s | 1.94s | +0.23s |
| **Tỷ lệ đồng thuận giữa các Judge** | 87.0% | 82.0% | -5.0% |
| **Tổng chi phí vận hành** | $0.0050 | $0.0070 | +$0.002 |

**Ghi chú kỹ thuật:**
- **Judges sử dụng:** Hệ thống đánh giá dựa trên sự kết hợp giữa kiến trúc Multi-Judge bao gồm OpenAI gpt-4o-mini và Deepseek deepseek-chat.
- **Cơ chế giải quyết xung đột:** Trong trường hợp hai Judge có sự chênh lệch điểm số lớn hơn 1, hệ thống sẽ ưu tiên lấy mức điểm thấp hơn để đảm bảo tính khách quan và an toàn (phương pháp đánh giá bảo thủ).
- **Cải tiến V2:** Tập trung vào việc tăng tham số top_k từ 3 lên 5, giúp cải thiện khả năng thu thập thông tin ngữ cảnh nhưng đồng thời làm tăng nhẹ độ trễ và chi phí.

---

## 2. Phân nhóm lỗi (Failure Clustering)

Việc phân tích sâu vào các nhóm lỗi giúp xác định chính xác các điểm nghẽn trong quy trình RAG hiện tại.

### 2.1 Hiệu suất theo mức độ khó của câu hỏi

Hệ thống cho thấy sự phân hóa rõ rệt về khả năng xử lý dựa trên độ phức tạp của truy vấn:

| Mức độ khó | Số lượng | Điểm TB | Tỷ lệ Pass | Nhận xét chi tiết |
|:---|:---:|:---:|:---:|:---|
| Easy | 10 | 4.55 | 100% | Hoàn thành xuất sắc các câu hỏi truy xuất trực tiếp dữ liệu. |
| Tricky (Adversarial) | 11 | 3.91 | 82% | System Prompt đã hoạt động hiệu quả trong việc ngăn chặn các hành vi tấn công prompt. |
| Hard (Edge Case) | 10 | 3.55 | 70% | Xử lý tốt các tình huống nằm ngoài phạm vi tài liệu (Out-of-Context). |
| Medium | 10 | 3.05 | 50% | **Điểm yếu cốt lõi:** Các câu hỏi yêu cầu kết hợp thông tin từ nhiều nguồn khác nhau. |
| Very Hard | 9 | 2.61 | 44% | Khả năng suy luận đa bước (Multi-fact reasoning) còn nhiều hạn chế. |

### 2.2 Phân tích các loại hình lỗi phổ biến

| Nhóm lỗi | Số lượng | Nguyên nhân gốc rễ dự kiến |
|:---|:---:|:---|
| **Truy xuất sai (Wrong Retrieval)** | 11 | Sự khác biệt về từ khóa giữa câu hỏi và tài liệu (Keyword Mismatch). |
| **Câu trả lời thiếu (Incomplete)** | 7 | Agent chỉ tập trung vào một phần nội dung trong các truy vấn đa mục tiêu. |
| **Lỗi âm tính giả (False Negative)** | 3 | Agent báo cáo không tìm thấy thông tin dù dữ liệu thực tế tồn tại trong tài liệu. |
| **Ảo tưởng (Hallucination)** | 2 | Agent tự tạo ra các thông tin không có thực (ví dụ: thông tin về số tín chỉ). |
| **Từ chối hợp lệ** | 1 | Agent từ chối trả lời các yêu cầu không an toàn (đây được tính là Pass). |

---

## 3. Phân tích nguyên nhân gốc rễ (5 Whys Analysis)

Chúng tôi thực hiện phân tích 5 Whys cho 3 trường hợp có điểm số thấp nhất để tìm giải pháp khắc phục triệt để.

### Trường hợp #1: Câu hỏi về "Nguyên tắc đánh giá kết quả học tập" (Score: 1/5)

1. **Triệu chứng (Symptom):** Agent phản hồi rằng thông tin không tồn tại trong tài liệu dù thực tế có quy định cụ thể.
2. **Tại sao 1?** Mô hình ngôn ngữ lớn (LLM) không nhận được thông tin liên quan trong context được cung cấp.
3. **Tại sao 2?** Bộ phận truy xuất (Retriever) đã trả về các đoạn văn bản (chunks) không liên quan, ví dụ như về "ngành đào tạo".
4. **Tại sao 3?** Có sự chênh lệch lớn về thuật ngữ (Keyword overlap thấp). Câu hỏi dùng từ "nguyên tắc", trong khi tài liệu dùng "hình thức" hoặc "phương thức".
5. **Tại sao 4?** Chiến lược phân tách dữ liệu (Chunking strategy) dựa hoàn toàn trên các tiêu đề bài viết, không dựa trên ngữ nghĩa.
6. **Nguyên nhân gốc rễ (Root Cause):** **Khoảng cách về cách diễn đạt (Paraphrasing gap).** Hệ thống truy xuất dựa trên từ khóa không thể nhận diện các khái niệm tương đồng nhưng dùng từ ngữ khác nhau. Cần áp dụng Vector Semantic Embedding.

### Trường hợp #2: Câu hỏi so sánh giữa "Học phần bắt buộc và tự chọn" (Score: 1/5)

1. **Triệu chứng:** Agent trả lời lạc đề, không so sánh được hai khái niệm này.
2. **Tại sao 1?** Ngữ cảnh truyền vào LLM thiếu các đoạn định nghĩa về hai loại học phần này.
3. **Tại sao 2?** Retriever ưu tiên các chunk có từ "chương trình đào tạo" hơn là tập trung vào từng loại "học phần".
4. **Tại sao 3?** Với câu hỏi so sánh, retriever bị nhầm lẫn và ưu tiên các đoạn văn bản mang tính giới thiệu tổng quát.
5. **Tại sao 4?** Thiếu cơ chế mở rộng truy vấn (Query Expansion) hoặc bản đồ từ đồng nghĩa.
6. **Nguyên nhân gốc rễ:** **Độ chính xác của việc truy xuất thấp (Retrieval Precision).** Cần triển khai kỹ thuật Query Rewriting hoặc HyDE (Hypothetical Document Embedding).

### Trường hợp #3: Điều 2 về "Tính chất của chương trình đào tạo" (Score: 1/5)

1. **Triệu chứng:** Câu trả lời không đề cập đúng các tính chất được nêu cụ thể tại Điều 2.
2. **Tại sao 1?** LLM không có đủ thông tin chi tiết về nội dung của Điều 2.
3. **Tại sao 2?** Hệ thống trả về thông tin của Điều 3 hoặc Điều 4 thay vì chính xác là Điều 2.
4. **Tại sao 3?** Câu hỏi có chứa số "2" nhưng hệ thống không sử dụng thông tin này để lọc dữ liệu (Metadata Filtering).
5. **Tại sao 4?** Quy trình chunking làm vỡ cấu trúc của từng điều luật, khiến thông tin bị phân tán.
6. **Nguyên nhân gốc rễ:** **Thiếu khả năng truy xuất dựa trên siêu dữ liệu (Metadata-aware retrieval).** Cần gắn thông tin số điều vào metadata và áp dụng kỹ thuật Parent-Child Chunking.

---

## 4. Kế hoạch hành động cải tiến (Action Plan)

Dựa trên kết quả phân tích, chúng tôi đề xuất các hành động cụ thể để cải thiện hệ thống:

| Ưu tiên | Hành động kỹ thuật | Tác động dự kiến | Mức độ thực hiện |
|:---:|:---|:---|:---:|
| **P0** | Thay thế truy xuất từ khóa bằng **text-embedding-3-small** | Tăng Hit Rate lên trên 85% | Trung bình |
| **P0** | Tích hợp định danh số điều vào Metadata và kích hoạt Filtering | Giải quyết triệt để lỗi truy xuất theo mã hiệu | Thấp |
| **P1** | Triển khai mô hình **Parent-Child Chunking** | Nâng cao khả năng thu hồi ngữ cảnh đầy đủ | Trung bình |
| **P1** | Áp dụng công nghệ **Query Rewriting** | Khắc phục vấn đề Paraphrasing Gap | Thấp |
| **P2** | Áp dụng **Reranking** (Cross-Encoder) sau bước truy xuất | Tối ưu hóa thứ tự ưu tiên của các chunk | Cao |
| **P2** | Tối ưu hóa **Two-stage judging** | Giảm thiểu chi phí đánh giá mà không mất dữ liệu | Thấp |

---

## 5. Đề xuất tối ưu hóa chi phí (Cost Optimization)

Giải pháp **Two-stage judging** được đề xuất nhằm giảm thiểu chi phí API:
- **Giai đoạn 1:** Sử dụng duy nhất mô hình Deepseek (chi phí thấp) để đánh giá sơ bộ toàn bộ 50 cases.
- **Giai đoạn 2:** Chỉ gửi các trường hợp có điểm số nằm trong vùng không chắc chắn (ví dụ: điểm 2 hoặc 3) tới quy trình Multi-Judge cấp cao hơn.
- **Hiệu quả:** Dự kiến giảm khoảng 33% tổng chi phí đánh giá (từ $0.007 xuống còn khoảng $0.0047) trong khi vẫn duy trì được độ chính xác tương đương cho các trường hợp rõ ràng.
