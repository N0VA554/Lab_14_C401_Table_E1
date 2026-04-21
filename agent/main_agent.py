import asyncio
import json
import os
from pathlib import Path
from typing import List, Dict
from openai import AsyncOpenAI
from dotenv import load_dotenv

load_dotenv()


class MainAgent:
    """
    RAG Agent: keyword-based retrieval from chunks.jsonl + OpenAI generation.
    top_k controls how many chunks are retrieved (V1=3, V2=5).
    """

    def __init__(self, top_k: int = 3):
        self.name = f"SupportAgent-top{top_k}"
        self.top_k = top_k
        self.client = AsyncOpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        self.chunks = self._load_chunks()
        self.total_tokens = 0
        self.total_cost_usd = 0.0

    def _load_chunks(self) -> List[Dict]:
        chunks_path = Path(__file__).parent.parent / "data" / "chunks.jsonl"
        chunks = []
        if not chunks_path.exists():
            return chunks
        with open(chunks_path, "r", encoding="utf-8") as f:
            for line in f:
                if line.strip():
                    chunks.append(json.loads(line))
        return chunks

    def _retrieve(self, question: str) -> List[Dict]:
        """Keyword overlap retrieval — simple but explainable."""
        q_words = set(question.lower().split())
        scored = []
        for chunk in self.chunks:
            chunk_words = set(chunk["text"].lower().split())
            score = len(q_words & chunk_words)
            scored.append((score, chunk))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [c for _, c in scored[: self.top_k]]

    async def query(self, question: str) -> Dict:
        retrieved = self._retrieve(question)
        context = "\n\n".join(c["text"] for c in retrieved)
        retrieved_ids = [c["chunk_id"] for c in retrieved]

        response = await self.client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Bạn là trợ lý AI chuyên trả lời câu hỏi dựa trên tài liệu quy định đào tạo. "
                        "Chỉ trả lời dựa trên context được cung cấp. "
                        "Nếu context không có thông tin, hãy nói rõ 'Thông tin này không có trong tài liệu.'"
                    ),
                },
                {
                    "role": "user",
                    "content": f"Context:\n{context}\n\nCâu hỏi: {question}",
                },
            ],
            temperature=0.1,
            max_tokens=500,
        )

        answer = response.choices[0].message.content.strip()
        usage = response.usage
        prompt_cost = usage.prompt_tokens * 0.00000015
        completion_cost = usage.completion_tokens * 0.0000006
        call_cost = prompt_cost + completion_cost

        self.total_tokens += usage.total_tokens
        self.total_cost_usd += call_cost

        return {
            "answer": answer,
            "contexts": [c["text"] for c in retrieved],
            "retrieved_ids": retrieved_ids,
            "metadata": {
                "model": "gpt-4o-mini",
                "tokens_used": usage.total_tokens,
                "cost_usd": round(call_cost, 6),
                "sources": retrieved_ids,
            },
        }


if __name__ == "__main__":
    async def _test():
        agent = MainAgent(top_k=3)
        resp = await agent.query("Chương trình đào tạo đại học tối thiểu bao nhiêu tín chỉ?")
        print(resp["answer"])
        print("Retrieved:", resp["retrieved_ids"])

    asyncio.run(_test())
