# Individual Reflection Report — Optimizer Role

| Thông tin | Chi tiết |
|---|---|
| **Họ và tên** | Nguyễn Đức Mạnh |
| **MSSV** | 2A202600151 |
| **Vai trò** | Optimizer (Agent Tuning & Cost) |
| **Ngày nộp** | 21/04/2026 |

---

## 1. Nhiệm vụ đảm nhận

Với vai trò **Optimizer**, tôi chịu trách nhiệm:
- Phân tích kết quả benchmark để xác định điểm yếu của Agent.
- Tối ưu hóa chiến lược retrieval và prompt của Agent.
- Thiết kế giải pháp **giảm chi phí Eval ≥ 30%** mà không giảm chất lượng.
- So sánh V1 vs V2 và đưa ra đề xuất cải tiến có căn cứ số liệu.

---

## 2. Những thay đổi đã thực hiện

### 2.1 Nâng cấp `agent/main_agent.py` — Real RAG Agent

Thay toàn bộ mock agent bằng RAG agent thực tế:

| Hạng mục | Trước (Mock) | Sau (Thực tế) |
|---|---|---|
| Retrieval | Hardcoded string | Keyword overlap trên 113 chunks |
| Generation | Không có LLM | OpenAI gpt-4o-mini |
| Cost tracking | Không có | Theo dõi token + USD từng call |
| top_k | N/A | Tham số cấu hình (V1=3, V2=5) |

**Cơ chế retrieval:**
```python
def _retrieve(self, question: str) -> List[Dict]:
    q_words = set(question.lower().split())
    scored = [(len(q_words & set(c["text"].lower().split())), c) for c in self.chunks]
    scored.sort(key=lambda x: x[0], reverse=True)
    return [c for _, c in scored[:self.top_k]]
```
Đây là keyword overlap retrieval — đơn giản, minh bạch, dễ debug và không cần embedding API.

### 2.2 Chiến lược V1 vs V2 — Tăng top_k từ 3 → 5

| Metric | V1 (top_k=3) | V2 (top_k=5) | Delta |
|---|---|---|---|
| Avg Judge Score | 3.250 | **3.560** | +0.310 🟢 |
| Pass Rate | 62.0% | **72.0%** | +10.0% 🟢 |
| Hit Rate | 62.0% | 62.0% | 0.0% |
| Avg MRR | 0.453 | **0.468** | +0.015 🟢 |
| Avg Latency | 1.71s | 1.94s | +0.23s 🟡 |
| Agreement Rate | 87.0% | 82.0% | -5.0% |

**Kết quả:** V2 được **APPROVE** bởi Regression Gate. Tăng top_k từ 3→5 cải thiện đáng kể Answer Quality (+9.5% score) và Pass Rate (+10%) với chi phí chỉ tăng $0.002.

### 2.3 Tối ưu hóa chi phí Eval

#### Giải pháp đã triển khai — Cost per eval: $0.000120

| Lựa chọn | Lý do |
|---|---|
| **gpt-4o-mini** thay vì gpt-4o | 20× rẻ hơn, chất lượng đủ cho scoring 1-5 |
| **Deepseek-chat** làm Judge 2 | ~10× rẻ hơn GPT-4o, agreement rate 82% |
| **Async parallel** judge calls | 2 judges chạy đồng thời, không tuần tự |
| **batch_size=5** | Tránh rate limit mà vẫn tận dụng concurrency |

#### Bảng so sánh chi phí thực tế

| Phương án | Cost/50 cases | Cost/1000 cases | Tiết kiệm |
|---|---|---|---|
| GPT-4o × 2 judges (baseline giả định) | ~$0.40 | ~$8.00 | — |
| **GPT-4o-mini + Deepseek (thực tế)** | **$0.012** | **$0.24** | **97%** |
| Chỉ dùng Deepseek × 2 | ~$0.004 | ~$0.08 | Thêm 67% |

> 💡 **Đề xuất giảm thêm 30% chi phí không giảm chất lượng:**
> Áp dụng **Two-stage filtering**: lần đầu chỉ chạy 1 judge (Deepseek) trên toàn bộ dataset, chỉ escalate các case có score 2-3 (biên giới pass/fail) lên Multi-Judge. Với phân phối hiện tại (~30% case biên giới), tổng chi phí judge giảm từ $0.012 → ~$0.008 (-33%), trong khi accuracy không đổi vì chỉ case uncertain mới cần 2 judges.

### 2.4 System Prompt Engineering

Thêm hướng dẫn rõ ràng khi không có context:
```
"Nếu context không có thông tin, hãy nói rõ 'Thông tin này không có trong tài liệu.'"
```
Điều này giảm hallucination cho Hard/OOC cases (điểm trung bình Hard: 3.55/5).

---

## 3. Phân tích kết quả theo độ khó

| Difficulty | Avg Score | Nhận xét |
|---|---|---|
| Easy | **4.55** | Agent xử lý tốt câu hỏi trực tiếp |
| Tricky (Adversarial) | **3.91** | System prompt phòng ngừa prompt injection tốt |
| Hard | 3.55 | OOC cases xử lý được nhờ "không biết" |
| Medium | 3.05 | Câu hỏi kết hợp nhiều section — điểm yếu chính |
| Very Hard | 2.61 | Multi-fact reasoning cần cải thiện |

**Điểm yếu nhất:** Các câu hỏi **Medium** yêu cầu kết hợp thông tin từ nhiều điều khoản khác nhau. Keyword retrieval không đủ mạnh để tìm đúng chunk khi câu hỏi dùng từ ngữ khác với tài liệu (paraphrasing gap).

---

## 4. Root Cause phân tích điểm thấp

**3 cases tệ nhất (score=1/5):**
1. "Nguyên tắc đánh giá kết quả học tập... có điểm gì đặc biệt?"
2. "So sánh học phần bắt buộc và học phần tự chọn..."
3. "Chương trình đào tạo đảm bảo tính cơ bản, thiết thực, hiện đại..."

**Root cause chung:** Agent trả lời rằng "thông tin không có trong tài liệu" mặc dù thực tế có — do keyword overlap retrieval không match được vì câu hỏi paraphrase so với nội dung chunk. Cụ thể: câu hỏi dùng "đặc biệt", "nguyên tắc" nhưng chunk chứa "quy định", "phương thức".

**Giải pháp đề xuất:** Thay keyword retrieval bằng **semantic embedding** (text-embedding-3-small) — chi phí chỉ tăng ~$0.001/50 cases nhưng giải quyết được paraphrasing gap.

---

## 5. Kết luận & Học được

1. **Cost optimization không đồng nghĩa với chất lượng thấp** — chọn đúng model cho đúng task (gpt-4o-mini + Deepseek đủ cho scoring, không cần flagship models).
2. **Retrieval quality là nút cổ chai** — Hit Rate 62% cho thấy 38% case agent không tìm đúng chunk, dù có context đúng trong tài liệu. Đây là nguyên nhân chính gây điểm thấp ở Medium/Very Hard.
3. **Async pipeline là bắt buộc** — 50 cases × 2 versions × 2 judges = 200 API calls hoàn thành trong < 4 phút nhờ asyncio.gather, nếu tuần tự sẽ mất > 30 phút.
4. **Two-stage judging** là chiến lược tối ưu cho production: tiết kiệm thêm ~33% chi phí mà không ảnh hưởng accuracy.
