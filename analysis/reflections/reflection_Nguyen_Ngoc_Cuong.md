# Individual Reflection Report — Optimizer Role

| Thông tin | Chi tiết |
|---|---|
| **Họ và tên** | Nguyễn Ngọc Cường |
| **MSSV** | 2A202600186 |
| **Vai trò** | AI lead (llm_judge) |
| **Ngày nộp** | 21/04/2026 |

---

## 1. Nhiệm vụ đảm nhận

Với vai trò **AI lead**, tôi chịu trách nhiệm:
- Xây dựng các hệ thống đánh giá cho module sinh câu trả lời (Generative Evaluation với Multi-LLM Judge) và module truy xuất (Retrieval Evaluation).
- Cài đặt cơ chế kiểm tra đa chiều như tỷ lệ đồng thuận, kiểm tra và khắc phục sự thiên vị vị trí (position bias) để kết quả đánh giá công bằng, minh bạch.
- Đảm bảo tính toán chính xác các số liệu truy xuất như Hit Rate và MRR.

## 2. Những thay đổi đã thực hiện

- **Tại `engine/llm_judge.py`**:
  - **Triển khai `LLMJudge` framework**: Tích hợp hai mô hình đánh giá gồm `gpt-4o-mini` (OpenAI) và `deepseek-chat` (Deepseek) thông qua API `AsyncOpenAI`. Xây dựng hệ thống prompt kỹ lưỡng yêu cầu chấm điểm câu trả lời trên thang điểm 1-5 và trả định dạng JSON.
  - **Xử lý Consensus (Đồng thuận)**: Thiết kế hàm `evaluate_multi_judge` để kiểm soát và tính tỷ lệ đồng thuận dựa trên độ chênh lệch. Nếu chênh lệch 0 điểm, `agreement_rate` là 1.0; lệch 1 điểm là 0.5 (ghi nhận điểm trung bình); lệch lớn hơn 1 điểm là 0.0 (xem như mâu thuẫn hệ thống, tự động sử dụng điểm thấp hơn để đảm bảo tính khắt khe).
  - **Kiểm tra Position Bias (Thiên vị vị trí)**: Xây dựng hàm `check_position_bias` để kiểm tra thiên vị mô hình bằng cách tráo đổi vị trí của hai đáp án. Thuật toán này phát hiện xem các mô hình có dính phải lỗi thiên vị đáp án đưa vào trước hay đứng sau không.

- **Tại `engine/retrieval_eval.py`**:
  - **Hoàn thiện các hàm đánh giá truy xuất**:
    - Cài đặt `calculate_hit_rate`: Tính toán xem văn bản có trong Top K `retrieved_ids` dựa trên danh sách `expected_ids` gốc không.
    - Cài đặt `calculate_mrr`: Lập trình tính thuật toán Mean Reciprocal Rank (tính nghịch đảo thứ hạng đầu tiên của kết quả chính xác xuất hiện trong danh sách).
  - **Xử lý đánh giá theo Batch**: Cập nhật hàm `evaluate_batch` duyệt qua mọi mẫu dữ liệu để tổng kết trung bình các chỉ số (`avg_hit_rate` và `avg_mrr`).

## 3. Technical Depth

**Giải thích các khái niệm đánh giá:**
- **MRR (Mean Reciprocal Rank)**: Là trung bình cộng nghịch đảo của thứ hạng (rank) mà tại đó văn bản mục tiêu xuất hiện đầu tiên trong danh sách truy xuất. MRR cho điểm càng cao nếu kết quả đúng xuất hiện càng gần top 1 (rank cao nhất là 1 điểm, rank 2 là 0.5 điểm...). Trong `retrieval_eval.py`, thuật toán chia `1.0 / (i + 1)` để xác định.
- **Cohen's Kappa (Agreement Rate)**: Là phép đo thống kê xem các giám khảo có nhất trí quan điểm với nhau không. Phiên bản tối giản cài đặt trong hệ thống trả về độ đồng thuận là `1.0` nếu bằng điểm, `0.5` nếu lệch 1 điểm, và `0.0` nếu đối kháng > 1 điểm, từ đó phát hiện các trường hợp mâu thuẫn đánh giá.
- **Position Bias (Thiên vị vị trí)**: Đây là một thiên vị phổ biến của LLM Judge khi nó có xu hướng ưu ái đáp án xuất hiện ở vị trí đầu tiên hoặc cuối cùng. Tính năng `check_position_bias` giải quyết triệt để lỗi này bằng cách yêu cầu LLM chấm điểm cả 2 chiều, tráo đổi chéo vị trí đáp án để đối chiếu sự kiên định của giám khảo.

**Hiểu về trade-off giữa Chi phí và Chất lượng:**
Trong hệ thống đánh giá đại lý AI, việc dùng mô hình rất lớn (như gpt-4o) đảm bảo độ chính xác cao nhưng chi phí token rất đắt đỏ để chạy trên bộ dữ liệu lớn. Ngược lại dùng mô hình quá nhỏ có thể dẫn tới sự đánh giá thiếu chất lượng. Hệ thống của tôi đã giải quyết tối ưu bài toán "Chi phí - Chất lượng" này bằng cách cấu trúc cơ chế **Multi-Judge** kết hợp 2 mô hình có chi phí cực thấp nhưng chất lượng tư duy ở mức tiệm cận: `gpt-4o-mini` và `deepseek-chat`. Việc cho hai giám khảo này kiểm tra chéo nhau sẽ phát hiện các ngoại lệ, từ đó mang lại độ tin cậy tương đương mô hình đắt tiền với một mức phí duy trì cực rẻ.

## 4. Problem Solving

**Cách giải quyết các vấn đề phát sinh trong quá trình xây dựng hệ thống phức tạp:**
- **Tránh nút thắt cổ chai độ trễ mạn:** Đánh giá bằng nhiều LLM cùng lúc đồng nghĩa hệ thống tốn vô số thời gian chờ I/O mạng. Tôi đã giải quyết vấn đề hiệu năng này bằng việc thiết kế toàn bộ luồng theo **Asynchronous Programing** và dùng `asyncio.gather()` để tận dụng tối đa lời gọi đồng thời đến OpenAI và Deepseek.
- **Kiểm soát tính ổn định cấu trúc Parsing:** Quá trình chấm điểm số lượng lớn rất dễ bị crash nếu LLM trả văn bản phi cấu trúc làm lỗi hàm đọc JSON. Giải pháp là ép hệ thống ràng buộc kiểu bằng cờ `response_format={"type": "json_object"}` và wrap bằng khối `try-except`, cung cấp ngay kết quả mặc định (fall-back) để không bao giờ làm gián đoạn pipeline.
- **Mâu thuẫn thông tin (Conflict Resolving):** Khi 2 API chấm chênh lệch nhau trên 1 điểm (ví dụ: mô hình A chấm 5, mô hình B chấm 2). Hệ thống được lập trình rẽ nhánh với chiến lược phạt nặng—tự động lấy điểm thấp hơn (`min(score_a, score_b)`) để đánh giá khắt khe nhất, bảo đảm không bị over-estimate chất lượng Agent.

## 5. ĐÓng góp commit
https://github.com/N0VA554/Lab_14_C401_Table_E1/commit/a5cf5ddbf397501ab1dff2191bf8665dff4ba3c5
https://github.com/N0VA554/Lab_14_C401_Table_E1/commit/e77516cfc94b1bcc98a558de428294d6c53be491
---
