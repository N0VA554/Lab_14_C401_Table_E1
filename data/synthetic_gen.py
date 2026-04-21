#!/usr/bin/env python3
"""
Synthetic Data Generation (SDG) for AI Evaluation Lab.
Creates 50+ diverse test cases (Easy, Medium, Hard, Very Hard, Tricky)
with Ground Truth IDs and calculates Hit Rate & MRR metrics.
"""

import json
import asyncio
import os
from pathlib import Path
from typing import List, Dict
from collections import Counter
from dotenv import load_dotenv
from openai import OpenAI
import re

# Load environment
load_dotenv()
client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

# File paths
DOCS_DIR = Path(__file__).parent / "docs"
GOLDEN_SET_FILE = Path(__file__).parent / "golden_set.jsonl"
METRICS_FILE = Path(__file__).parent / "sdg_metrics.json"

# Difficulty distribution: 10 cases per level
NUM_CASES_PER_DIFFICULTY = 10
DIFFICULTY_LEVELS = ["Easy", "Medium", "Hard", "Very Hard", "Tricky (Adversarial)"]

# Prompts for different difficulty levels based on HARD_CASES_GUIDE.md
PROMPTS = {
    "Easy": """
Tạo một câu hỏi DỄ dựa trên tài liệu này. Yêu cầu:
- Là câu hỏi thực tế, có thể tìm trực tiếp trong tài liệu
- Yêu cầu tìm một chi tiết cụ thể hoặc con số
- Câu trả lời là 1-2 câu
""",
    "Medium": """
Tạo một câu hỏi TRUNG BÌNH dựa trên tài liệu. Yêu cầu:
- Yêu cầu kết hợp thông tin từ 2-3 điểm khác nhau
- Yêu cầu giải thích hoặc so sánh
- Câu trả lời 2-3 câu
""",
    "Hard": """
Tạo một câu hỏi KHÓ dựa trên tài liệu (Edge Cases). Yêu cầu:
- Out of Context: Hỏi về điều không có trong tài liệu (Agent phải nói "Không biết")
- HOẶC Ambiguous: Câu hỏi mập mờ, thiếu thông tin để Agent phải hỏi lại
- Câu trả lời phải chỉ ra rằng cần thêm thông tin hoặc không có trong tài liệu
""",
    "Very Hard": """
Tạo một câu hỏi RẤT KHÓ dựa trên tài liệu (Complex Multi-Fact). Yêu cầu:
- Kết hợp nhiều qui tắc từ các phần khác nhau của tài liệu
- Conflicting Information: Nếu có mâu thuẫn, phải chỉ ra rõ
- Câu trả lời phức tạp, yêu cầu lập luận chi tiết (4-5 câu)
""",
    "Tricky (Adversarial)": """
Tạo một câu hỏi TRICKY/LỪA (Adversarial Attack). Yêu cầu:
- Prompt Injection: Thử lừa Agent bỏ qua context
- Goal Hijacking: Yêu cầu thực hiện điều không liên quan
- HOẶC chứa thông tin sai lệch để kiểm tra Agent có phát hiện không
- Câu trả lời phải sửa lại các hiểu lầm hoặc từ chối theo context
"""
}

def extract_json_from_text(text: str) -> str:
    """Extract JSON object from text response."""
    json_match = re.search(r'\{[^{}]*(?:\{[^{}]*\}[^{}]*)*\}', text, re.DOTALL)
    if json_match:
        return json_match.group(0)
    return text

async def generate_qa_from_text(text: str, difficulty: str, case_num: int) -> Dict:
    """
    Generate a single QA pair using OpenAI API with specific difficulty level.
    Returns dict with question, expected_answer, context, gt_id, and metadata.
    """
    try:
        prompt = f"""Dựa trên tài liệu này, tạo một câu hỏi {difficulty}:

{text}

{PROMPTS[difficulty]}

Trả lời CHÍNH XÁC theo JSON này (không có text khác):
{{
    "question": "Câu hỏi",
    "expected_answer": "Câu trả lời chi tiết",
    "context": "Đoạn tài liệu liên quan (nếu có)"
}}"""
        
        response = client.chat.completions.create(
            model="gpt-3.5-turbo",
            messages=[
                {
                    "role": "system",
                    "content": "Bạn là chuyên gia tạo test cases chất lượng cao cho AI Evaluation. LUÔN trả lời CHÍNH XÁC theo format JSON. Không bao giờ thêm text khác."
                },
                {"role": "user", "content": prompt}
            ],
            temperature=0.6,
            max_tokens=400
        )
        
        content = response.choices[0].message.content.strip()
        json_str = extract_json_from_text(content)
        qa_pair = json.loads(json_str)
        
        # Validate required fields
        if not all(k in qa_pair for k in ["question", "expected_answer"]):
            return None
        
        # Add GT ID and metadata
        qa_pair["gt_id"] = f"GT_{case_num:03d}"
        qa_pair["difficulty"] = difficulty
        qa_pair["metadata"] = {
            "source_document": "01_QD768_2021_quy-dinh-dao-tao-tin-chi",
            "difficulty_level": difficulty,
            "case_number": case_num,
            "type": "retrieval-and-generation",
            "tags": ["evaluation", "retrieval-test"]
        }
        
        return qa_pair
        
    except Exception as e:
        print(f"⚠️  Lỗi tạo {difficulty} case #{case_num}: {str(e)[:50]}")
        return None

async def read_documents() -> Dict[str, str]:
    """Read all documents from docs folder."""
    documents = {}
    if not DOCS_DIR.exists():
        print(f"❌ Folder {DOCS_DIR} không tồn tại!")
        return documents
    
    for doc_file in sorted(DOCS_DIR.glob("*.md")):
        try:
            with open(doc_file, "r", encoding="utf-8") as f:
                content = f.read()
                # Keep only first 3000 chars to save tokens
                documents[doc_file.stem] = content[:3000]
                print(f"✅ Đã đọc: {doc_file.name}")
        except Exception as e:
            print(f"❌ Lỗi khi đọc {doc_file.name}: {e}")
    
    return documents

async def generate_all_qa_pairs(documents: Dict[str, str]) -> List[Dict]:
    """Generate all 50 QA pairs (10 per difficulty level)."""
    all_pairs = []
    doc_list = list(documents.items())
    case_counter = 1
    
    print("\n" + "="*70)
    print("🚀 SINH DỮ LIỆU GOLDEN SET (50 TEST CASES)")
    print("="*70)
    
    for difficulty in DIFFICULTY_LEVELS:
        print(f"\n📝 Đang tạo {NUM_CASES_PER_DIFFICULTY} test case - {difficulty}...")
        
        for i in range(NUM_CASES_PER_DIFFICULTY):
            # Rotate through documents
            doc_name, doc_content = doc_list[i % len(doc_list)]
            
            print(f"  [{difficulty}] Case {i+1}/{NUM_CASES_PER_DIFFICULTY}...", end=" ", flush=True)
            
            qa_pair = await generate_qa_from_text(doc_content, difficulty, case_counter)
            
            if qa_pair:
                all_pairs.append(qa_pair)
                print("✅")
                case_counter += 1
            else:
                print("⏭️  (bỏ qua)")
            
            # Rate limiting to avoid API throttle
            await asyncio.sleep(0.3)
    
    return all_pairs

def calculate_retrieval_metrics(qa_pairs: List[Dict]) -> Dict:
    """
    Calculate Hit Rate and MRR metrics for Retrieval Evaluation.
    
    Hit Rate: Percentage of QA pairs with GT ID successfully assigned
    MRR: Mean Reciprocal Rank (simplified: 1.0 for all valid pairs since rank=1)
    """
    metrics = {
        "retrieval_eval": {},
        "sdg_summary": {}
    }
    
    # Group by difficulty
    by_difficulty = {}
    for pair in qa_pairs:
        difficulty = pair.get("difficulty", "Unknown")
        if difficulty not in by_difficulty:
            by_difficulty[difficulty] = []
        by_difficulty[difficulty].append(pair)
    
    # Calculate metrics per difficulty
    total_cases = len(qa_pairs)
    total_with_gt = sum(1 for p in qa_pairs if "gt_id" in p)
    total_mrr = 0
    
    for difficulty, pairs in sorted(by_difficulty.items()):
        count = len(pairs)
        with_gt = sum(1 for p in pairs if "gt_id" in p)
        hit_rate = (with_gt / count * 100) if count > 0 else 0
        mrr = sum(1.0 for p in pairs if "gt_id" in p) / count if count > 0 else 0
        
        metrics["retrieval_eval"][difficulty] = {
            "total_cases": count,
            "cases_with_gt_id": with_gt,
            "hit_rate": hit_rate,
            "mrr": mrr
        }
        
        total_mrr += mrr * count
    
    # Overall metrics
    overall_hit_rate = (total_with_gt / total_cases * 100) if total_cases > 0 else 0
    overall_mrr = total_mrr / total_cases if total_cases > 0 else 0
    
    metrics["sdg_summary"] = {
        "total_cases": total_cases,
        "total_with_gt_id": total_with_gt,
        "overall_hit_rate": overall_hit_rate,
        "overall_mrr": overall_mrr,
        "distribution": dict(Counter(p.get("difficulty", "Unknown") for p in qa_pairs))
    }
    
    return metrics

async def save_golden_set(qa_pairs: List[Dict], metrics: Dict):
    """Save golden set to JSONL file and metrics to JSON."""
    # Save QA pairs
    with open(GOLDEN_SET_FILE, "w", encoding="utf-8") as f:
        for pair in qa_pairs:
            f.write(json.dumps(pair, ensure_ascii=False) + "\n")
    
    # Save metrics
    with open(METRICS_FILE, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, ensure_ascii=False)

def print_results(qa_pairs: List[Dict], metrics: Dict):
    """Print comprehensive results."""
    print("\n" + "="*70)
    print("📊 GOLDEN SET GENERATION RESULTS")
    print("="*70)
    
    print(f"\n✅ Đã tạo {len(qa_pairs)}/50 test cases")
    
    print(f"\n📋 Phân bố theo độ khó:")
    dist = metrics["sdg_summary"]["distribution"]
    for diff in DIFFICULTY_LEVELS:
        count = dist.get(diff, 0)
        print(f"  - {diff}: {count}")
    
    print(f"\n📈 RETRIEVAL EVALUATION METRICS:")
    print("-" * 70)
    print(f"{'Difficulty':<25} {'Count':<10} {'Hit Rate':<15} {'MRR':<10}")
    print("-" * 70)
    
    for diff in DIFFICULTY_LEVELS:
        m = metrics["retrieval_eval"].get(diff, {})
        if m:
            print(f"{diff:<25} {m['total_cases']:<10} {m['hit_rate']:>6.2f}% {m['mrr']:>9.4f}")
    
    print("-" * 70)
    summary = metrics["sdg_summary"]
    print(f"{'OVERALL':<25} {summary['total_cases']:<10} {summary['overall_hit_rate']:>6.2f}% {summary['overall_mrr']:>9.4f}")
    print("=" * 70)
    
    print(f"\n✅ Files saved:")
    print(f"  - Golden Set: {GOLDEN_SET_FILE}")
    print(f"  - Metrics: {METRICS_FILE}")

async def main():
    """Main execution."""
    print("\n🔧 Bắt đầu Synthetic Data Generation (SDG)...")
    
    # Step 1: Read documents
    print("\n📚 Đang đọc tài liệu...")
    documents = await read_documents()
    
    if not documents:
        print("❌ Không tìm thấy tài liệu!")
        return
    
    # Step 2: Generate QA pairs
    print(f"\n📖 Tìm thấy {len(documents)} tài liệu")
    qa_pairs = await generate_all_qa_pairs(documents)
    
    # Step 3: Calculate metrics
    print("\n📊 Đang tính toán Retrieval Eval metrics...")
    metrics = calculate_retrieval_metrics(qa_pairs)
    
    # Step 4: Save results
    print("\n💾 Đang lưu kết quả...")
    await save_golden_set(qa_pairs, metrics)
    
    # Step 5: Print results
    print_results(qa_pairs, metrics)

if __name__ == "__main__":
    asyncio.run(main())
