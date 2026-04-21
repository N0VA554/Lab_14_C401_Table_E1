import asyncio
import json
import os
from typing import Dict, Any
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


class LLMJudge:
    """
    Multi-judge consensus using OpenAI (gpt-4o-mini) and Deepseek (deepseek-chat).
    Computes agreement rate and handles score conflicts automatically.
    """

    def __init__(self):
        self.openai_client = AsyncOpenAI(
            api_key=os.environ.get("OPENAI_API_KEY", ""),
        )
        self.deepseek_client = AsyncOpenAI(
            api_key=os.environ.get("DEEPSEEK_API_KEY", ""),
            base_url="https://api.deepseek.com/v1",
        )

        self.score_prompt = """
Bạn là chuyên gia đánh giá AI. Hãy chấm điểm câu trả lời từ 1-5:
1 = Sai hoàn toàn
2 = Phần lớn sai hoặc thiếu thông tin quan trọng
3 = Đúng một phần, còn thiếu hoặc chưa chính xác
4 = Đúng phần lớn, thiếu ít chi tiết
5 = Chính xác và đầy đủ hoàn toàn

Chỉ trả về JSON: {{"score": <1-5>, "reasoning": "<giải thích ngắn gọn>"}}

Câu hỏi: {question}
Đáp án chuẩn: {ground_truth}
Câu trả lời hệ thống: {answer}
"""

        self.compare_prompt = """
Bạn là chuyên gia đánh giá. So sánh hai câu trả lời và chọn câu tốt hơn.
Chỉ trả về JSON: {{"better_response": 1}} hoặc {{"better_response": 2}} hoặc {{"better_response": 0}} (nếu bằng nhau).

Câu trả lời 1: {response_1}
Câu trả lời 2: {response_2}
"""

    async def _evaluate_single(
        self, client: AsyncOpenAI, model: str, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        prompt = self.score_prompt.format(
            question=question, ground_truth=ground_truth, answer=answer
        )
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            result = json.loads(resp.choices[0].message.content)
            return {
                "score": max(1, min(5, int(result.get("score", 1)))),
                "reasoning": result.get("reasoning", ""),
            }
        except Exception as e:
            print(f"  ⚠️  Judge {model} error: {e}")
            return {"score": 1, "reasoning": f"Error: {str(e)[:60]}"}

    async def evaluate_multi_judge(
        self, question: str, answer: str, ground_truth: str
    ) -> Dict[str, Any]:
        """
        Chạy song song 2 judges (OpenAI + Deepseek).
        Agreement rate: 1.0 nếu đồng điểm, 0.5 nếu lệch 1, 0.0 nếu lệch >1.
        Khi lệch >1: lấy điểm thấp hơn (đánh giá khắt khe).
        """
        openai_task = self._evaluate_single(
            self.openai_client, "gpt-4o-mini", question, answer, ground_truth
        )
        deepseek_task = self._evaluate_single(
            self.deepseek_client, "deepseek-chat", question, answer, ground_truth
        )

        res_a, res_b = await asyncio.gather(openai_task, deepseek_task)
        score_a, score_b = res_a["score"], res_b["score"]
        diff = abs(score_a - score_b)

        if diff == 0:
            final_score = score_a
            agreement = 1.0
        elif diff == 1:
            final_score = (score_a + score_b) / 2
            agreement = 0.5
        else:
            print(f"  ⚠️  Conflict: OpenAI={score_a}, Deepseek={score_b} — using lower score")
            final_score = float(min(score_a, score_b))
            agreement = 0.0

        return {
            "final_score": final_score,
            "agreement_rate": agreement,
            "reasoning": res_a["reasoning"],
            "individual_scores": {
                "openai-gpt4o-mini": score_a,
                "deepseek-v3": score_b,
            },
        }

    async def _compare_responses(
        self, client: AsyncOpenAI, model: str, response_1: str, response_2: str
    ) -> int:
        prompt = self.compare_prompt.format(response_1=response_1, response_2=response_2)
        try:
            resp = await client.chat.completions.create(
                model=model,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.0,
                response_format={"type": "json_object"},
                max_tokens=50,
            )
            result = json.loads(resp.choices[0].message.content)
            return int(result.get("better_response", 0))
        except Exception:
            return 0

    async def check_position_bias(
        self, response_a: str, response_b: str
    ) -> Dict[str, Any]:
        """
        Position bias test: swap A/B order and check if judge changes verdict.
        Applied to both OpenAI and Deepseek judges.
        """
        g1 = self._compare_responses(self.openai_client, "gpt-4o-mini", response_a, response_b)
        g2 = self._compare_responses(self.openai_client, "gpt-4o-mini", response_b, response_a)
        d1 = self._compare_responses(self.deepseek_client, "deepseek-chat", response_a, response_b)
        d2 = self._compare_responses(self.deepseek_client, "deepseek-chat", response_b, response_a)

        g_res1, g_res2, d_res1, d_res2 = await asyncio.gather(g1, g2, d1, d2)

        def _analyze(r1, r2):
            if r1 == 1 and r2 == 1:
                return {"has_bias": True, "bias_type": "Thiên vị vị trí đầu"}
            elif r1 == 2 and r2 == 2:
                return {"has_bias": True, "bias_type": "Thiên vị vị trí sau"}
            return {"has_bias": False, "bias_type": "Không phát hiện thiên vị"}

        return {
            "openai": _analyze(g_res1, g_res2),
            "deepseek": _analyze(d_res1, d_res2),
        }
