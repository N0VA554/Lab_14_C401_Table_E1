import asyncio
import json
import os
from typing import Dict, Any
from openai import AsyncOpenAI
import dotenv

dotenv.load_dotenv()

class LLMJudge:
    def __init__(self):
        # Khởi tạo 2 client thông qua chuẩn OpenAI tương thích
        self.openai_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", "dummy_key")
        )
        self.deepseek_client = AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", "dummy_key"),
            base_url="https://api.deepseek.com/v1"
        )
        
        self.compare_prompt_template = """
Bạn là một chuyên gia đánh giá khách quan. Hãy so sánh hai câu trả lời sau đây và chọn ra câu trả lời tốt hơn.
Chỉ trả về định dạng JSON duy nhất với cấu trúc: {{"better_response": 1}} hoặc {{"better_response": 2}} hoặc {{"better_response": 0}} (0 nếu hòa nhau).

Câu trả lời 1: {response_1}
Câu trả lời 2: {response_2}
"""

        self.prompt_template = """
Bạn là một chuyên gia đánh giá. Hãy đánh giá độ chính xác của câu trả lời dựa trên câu hỏi và đáp án tham chiếu.
Bạn phải chấm điểm từ 1 đến 5:
1 = Trả lời sai hoàn toàn
5 = Trả lời chính xác hoàn toàn

Chỉ trả về một đối tượng JSON với cấu trúc: {{"score": 4, "reasoning": "câu giải thích ngắn gọn"}}

Câu hỏi: {question}
Đáp án tham chiếu (Ground Truth): {ground_truth}
Câu trả lời của hệ thống: {answer}
"""

    async def _evaluate_single(self, client: AsyncOpenAI, model: str, question: str, answer: str, ground_truth: str) -> int:
        prompt = self.prompt_template.format(
            question=question, ground_truth=ground_truth, answer=answer
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            return int(result.get("score", 1))
        except Exception as e:
            print(f"Lỗi khi gọi model {model}: {e}")
            return 1 # Mặc định trả về 1 nếu gặp lỗi

    async def evaluate_multi_judge(self, question: str, answer: str, ground_truth: str) -> Dict[str, Any]:
        """
        EXPERT TASK: Gọi 2 model (GPT-3.5 Turbo và Deepseek V3).
        Tính toán sự sai lệch. Nếu lệch > 1 điểm, cần logic xử lý.
        """
        # Chạy song song 2 request
        openai_task = self._evaluate_single(
            self.openai_client, "gpt-3.5-turbo", question, answer, ground_truth
        )
        deepseek_task = self._evaluate_single(
            self.deepseek_client, "deepseek-chat", question, answer, ground_truth # deepseek-chat == V3
        )
        
        score_a, score_b = await asyncio.gather(openai_task, deepseek_task)
        
        avg_score = (score_a + score_b) / 2
        
        # Logic xử lý sai lệch lớn hơn 1 điểm
        if abs(score_a - score_b) > 1:
            print(f"⚠️ Cảnh báo: Các Giám khảo có độ lệch lớn (GPT-3.5: {score_a}, Deepseek: {score_b}).")
            # Nếu sai lệch > 1, lấy điểm thấp hơn để đánh giá khắt khe (hoặc có thể dùng LLM thứ 3 để trọng tài)
            avg_score = min(score_a, score_b)
            
        # Tính toán mức độ đồng thuận
        if score_a == score_b:
            agreement = 1.0
        elif abs(score_a - score_b) == 1:
            agreement = 0.5
        else:
            agreement = 0.0
        
        return {
            "final_score": avg_score,
            "agreement_rate": agreement,
            "individual_scores": {
                "gpt-3.5-turbo": score_a,
                "deepseek-v3": score_b
            }
        }

    async def _compare_responses(self, client: AsyncOpenAI, model: str, response_1: str, response_2: str) -> int:
        prompt = self.compare_prompt_template.format(
            response_1=response_1, response_2=response_2
        )
        try:
            response = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"}
            )
            content = response.choices[0].message.content
            result = json.loads(content)
            return int(result.get("better_response", 0))
        except Exception as e:
            print(f"Lỗi khi so sánh: {e}")
            return 0

    async def check_position_bias(self, response_a: str, response_b: str) -> Dict[str, Any]:
        """
        Nâng cao: Thực hiện đổi chỗ response A và B để xem Judge có thiên vị vị trí không.
        Áp dụng cho CẢ 2 mô hình (GPT-3.5 và Deepseek) để so sánh.
        """
        # Test GPT-3.5 Turbo
        g_task_1 = self._compare_responses(self.openai_client, "gpt-3.5-turbo", response_a, response_b)
        g_task_2 = self._compare_responses(self.openai_client, "gpt-3.5-turbo", response_b, response_a)
        
        # Test Deepseek
        d_task_1 = self._compare_responses(self.deepseek_client, "deepseek-chat", response_a, response_b)
        d_task_2 = self._compare_responses(self.deepseek_client, "deepseek-chat", response_b, response_a)
        
        g_res_1, g_res_2, d_res_1, d_res_2 = await asyncio.gather(
            g_task_1, g_task_2, d_task_1, d_task_2
        )
        
        def _analyze_bias(res_1, res_2):
            winner_1 = "A" if res_1 == 1 else ("B" if res_1 == 2 else "tie")
            winner_2 = "B" if res_2 == 1 else ("A" if res_2 == 2 else "tie")
            
            has_bias = False
            if res_1 == 1 and res_2 == 1:
                has_bias = True
                bias_type = "Thiên vị vị trí thứ nhất (Luôn chọn đáp án đầu)"
            elif res_1 == 2 and res_2 == 2:
                has_bias = True
                bias_type = "Thiên vị vị trí thứ hai (Luôn chọn đáp án sau)"
            else:
                bias_type = "Không phát hiện thiên vị vị trí"
                
            return {
                "test_1_winner": winner_1,
                "test_2_winner": winner_2,
                "has_bias": has_bias,
                "bias_type": bias_type
            }

        return {
            "gpt_3_5_turbo": _analyze_bias(g_res_1, g_res_2),
            "deepseek": _analyze_bias(d_res_1, d_res_2)
        }
