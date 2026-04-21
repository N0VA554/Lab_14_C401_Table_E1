from typing import List, Dict

class RetrievalEvaluator:
    def __init__(self):
        pass

    def calculate_hit_rate(self, expected_ids: List[str], retrieved_ids: List[str], top_k: int = 3) -> float:
        """
        TODO: Tính toán xem ít nhất 1 trong expected_ids có nằm trong top_k của retrieved_ids không.
        """
        top_retrieved = retrieved_ids[:top_k]
        hit = any(doc_id in top_retrieved for doc_id in expected_ids)
        return 1.0 if hit else 0.0

    def calculate_mrr(self, expected_ids: List[str], retrieved_ids: List[str]) -> float:
        """
        TODO: Tính Mean Reciprocal Rank.
        Tìm vị trí đầu tiên của một expected_id trong retrieved_ids.
        MRR = 1 / position (vị trí 1-indexed). Nếu không thấy thì là 0.
        """
        for i, doc_id in enumerate(retrieved_ids):
            if doc_id in expected_ids:
                return 1.0 / (i + 1)
        return 0.0

    async def evaluate_batch(self, dataset: List[Dict]) -> Dict:
        """
        Chạy eval cho toàn bộ bộ dữ liệu.
        Dataset cần có trường 'expected_retrieval_ids' và Agent trả về 'retrieved_ids'.
        """
        if not dataset:
            return {"avg_hit_rate": 0.0, "avg_mrr": 0.0}
            
        total_hit_rate = 0.0
        total_mrr = 0.0
        n_samples = len(dataset)
        
        for item in dataset:
            # Lấy ra danh sách các document ID dự kiến và thực tế (nếu không có thì trả về mảng rỗng)
            expected = item.get("expected_retrieval_ids", [])
            retrieved = item.get("retrieved_ids", [])
            
            # Cộng dồn điểm cho từng dòng dữ liệu
            total_hit_rate += self.calculate_hit_rate(expected, retrieved)
            total_mrr += self.calculate_mrr(expected, retrieved)
            
        # Tính điểm trung bình toàn bộ Dataset
        avg_hit_rate = total_hit_rate / n_samples
        avg_mrr = total_mrr / n_samples
        
        return {
            "avg_hit_rate": avg_hit_rate,
            "avg_mrr": avg_mrr
        }
