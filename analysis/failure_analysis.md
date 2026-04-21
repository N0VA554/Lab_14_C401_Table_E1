# Báo cáo Phân tích Thất bại (Failure Analysis Report)

## 1. Tổng quan Benchmark

| Chỉ số | V1 (Base) | V2 (Optimized) |
|--------|-----------|----------------|
| **Tổng số cases** | 50 | 50 |
| **Pass / Fail** | 31 / 19 | **36 / 14** |
| **Pass Rate** | 62.0% | **72.0%** |
| **Avg Judge Score** | 3.250 / 5.0 | **3.560 / 5.0** |
| **Hit Rate (Retrieval)** | 62.0% | 62.0% |
| **MRR** | 0.453 | 0.468 |
| **Avg Latency** | 1.71s | 1.94s |
| **Agreement Rate (2 judges)** | 87.0% | 82.0% |
| **Total Cost** | $0.0050 | $0.0070 |

**Judges sử dụng:** OpenAI gpt-4o-mini + Deepseek deepseek-chat  
**Conflict resolution:** Khi 2 judges lệch > 1 điểm → lấy điểm thấp hơn (conservative)

---

## 2. Phân nhóm lỗi (Failure Clustering)

### 2.1 Phân bố điểm theo độ khó

| Difficulty | N | Avg Score | Pass Rate | Nhận xét |
|---|---|---|---|---|
| Easy | 10 | 4.55 | 100% | Tốt — câu hỏi trực tiếp |
| Tricky (Adversarial) | 11 | 3.91 | 82% | System prompt phòng ngừa tốt |
| Hard (Edge Case) | 10 | 3.55 | 70% | OOC xử lý được |
| Medium | 10 | 3.05 | 50% | **Điểm yếu chính** |
| Very Hard | 9 | 2.61 | 44% | Multi-fact reasoning yếu |

### 2.2 Phân loại nhóm lỗi

| Nhóm lỗi | Số cases | Nguyên nhân gốc |
|---|---|---|
| **Wrong Retrieval** | 11 | Keyword mismatch — paraphrase gap giữa câu hỏi và chunk |
| **Incomplete Answer** | 7 | Agent chỉ trả lời 1 trong nhiều yêu cầu của câu hỏi phức tạp |
| **False Negative (OOC)** | 3 | Agent nói "không có trong tài liệu" dù thực tế có |
| **Hallucination** | 2 | Agent bịa số liệu tín chỉ không đúng |
| **Refused Adversarial** | 1 | Agent từ chối hợp lệ (prompt injection) — đây là PASS |

---

## 3. Phân tích 5 Whys (3 case tệ nhất)

### Case #1: "Nguyên tắc đánh giá kết quả học tập có điểm gì đặc biệt?" (Score: 1/5)

1. **Symptom:** Agent trả lời "Thông tin này không có trong tài liệu" — nhưng thực tế có.
2. **Why 1:** LLM không thấy thông tin liên quan trong context được truyền vào.
3. **Why 2:** Retriever trả về sai chunks — chunks về "ngành đào tạo" thay vì "đánh giá kết quả".
4. **Why 3:** Keyword overlap giữa câu hỏi ("nguyên tắc", "đánh giá") và chunk target thấp — chunk dùng từ "hình thức kiểm tra", "thi kết thúc học phần".
5. **Why 4:** Chunking strategy chia tài liệu theo heading, không theo semantic topic.
6. **Root Cause:** **Paraphrasing gap** — Retrieval dựa trên keyword exact match không xử lý được khi câu hỏi và tài liệu dùng từ ngữ khác nhau để diễn đạt cùng khái niệm. Cần semantic embedding.

---

### Case #2: "So sánh học phần bắt buộc và học phần tự chọn..." (Score: 1/5)

1. **Symptom:** Agent cung cấp câu trả lời hoàn toàn không liên quan đến nội dung câu hỏi.
2. **Why 1:** Context đưa vào LLM không chứa đoạn định nghĩa "học phần bắt buộc" và "học phần tự chọn".
3. **Why 2:** Retriever ưu tiên chunks chứa từ "chương trình đào tạo" thay vì "học phần".
4. **Why 3:** Câu hỏi dùng từ "so sánh" → retriever score cao cho chunks chứa nhiều từ overlap với phần giới thiệu tổng quát.
5. **Why 4:** Không có query expansion hay synonym mapping ("học phần bắt buộc" ↔ "compulsory course").
6. **Root Cause:** **Retrieval precision thấp** với câu hỏi so sánh — cần implement **query rewriting** hoặc **HyDE (Hypothetical Document Embedding)** để cải thiện.

---

### Case #3: "Chương trình đào tạo đảm bảo tính cơ bản, thiết thực, hiện đại theo Điều 2?" (Score: 1/5)

1. **Symptom:** Câu trả lời không đề cập đến Điều 2, không nêu được 3 tính chất.
2. **Why 1:** LLM không có đủ context về Điều 2 cụ thể.
3. **Why 2:** Retriever trả về chunk của Điều 3, Điều 4 thay vì Điều 2.
4. **Why 3:** Câu hỏi đề cập "Điều 2" nhưng chunk_id không encode số điều — không có metadata filtering.
5. **Why 4:** Chunking không giữ nguyên cấu trúc Điều → các mục của Điều 2 bị chia vào nhiều chunk nhỏ, mỗi chunk không đủ score để rank cao.
6. **Root Cause:** **Thiếu metadata-aware retrieval** — cần lưu `article_number` vào chunk metadata để enable filtering; đồng thời **parent-child chunking** để khi match được mục nhỏ vẫn retrieve được toàn bộ điều.

---

## 4. Kế hoạch cải tiến (Action Plan)

| Ưu tiên | Hành động | Impact dự kiến | Effort |
|---|---|---|---|
| 🔴 P0 | Thay keyword retrieval bằng **text-embedding-3-small** | Hit Rate: 62% → ~85% | Medium |
| 🔴 P0 | Thêm `article_number` vào chunk metadata, enable filtering | Giải quyết Case #3 | Low |
| 🟡 P1 | Implement **parent-child chunking** (child chunk → parent article) | Cải thiện recall cho câu hỏi phức tạp | Medium |
| 🟡 P1 | **Query rewriting**: paraphrase câu hỏi thành nhiều variant trước khi retrieve | Giải quyết paraphrasing gap | Low |
| 🟢 P2 | **Two-stage judging**: Deepseek-only cho cases dễ, Multi-judge cho biên giới | Giảm 33% cost judge | Low |
| 🟢 P2 | Thêm **reranking** (cross-encoder) sau retrieval | Cải thiện precision của top-3 | High |

### Cost Optimization đề xuất

Áp dụng **Two-stage judging**:
- Stage 1: Chỉ dùng Deepseek (rẻ) cho tất cả 50 cases.
- Stage 2: Chỉ escalate ~30% cases có score biên giới (2-3) lên Multi-Judge.
- **Kết quả: giảm ~33% chi phí judge** từ $0.007 → ~$0.0047 mà không giảm accuracy với cases rõ ràng.

```
# Ví dụ logic two-stage
if deepseek_score in [2, 3]:   # uncertain → dùng cả 2 judges
    final = await multi_judge(q, a, gt)
else:                           # rõ ràng → tin vào Deepseek
    final = deepseek_score
```
