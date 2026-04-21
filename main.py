import asyncio
import json
import os
import time
from datetime import datetime

# --- MOCK COMPONENTS (Thay bằng engine thực tế của bạn) ---

class MockMainAgent:
    async def run(self, query):
        await asyncio.sleep(0.1) 
        return f"Câu trả lời cho: {query}"

class ExpertEvaluator:
    async def score(self, case, resp): 
        return {
            "faithfulness": 0.9, 
            "relevancy": 0.8,
            "retrieval": {"hit_rate": 1.0, "mrr": 0.5}
        }

class MultiModelJudge:
    async def evaluate_multi_judge(self, q, a, gt): 
        return {
            "final_score": 4.5, 
            "agreement_rate": 0.8,
            "reasoning": "Câu trả lời đạt yêu cầu."
        }

class BenchmarkRunner:
    def __init__(self, agent, expert, judge):
        self.agent = agent
        self.expert = expert
        self.judge = judge

    async def run_all(self, dataset):
        tasks = [self._run_single_case(case) for case in dataset]
        return await asyncio.gather(*tasks)

    async def _run_single_case(self, case):
        start_time = time.perf_counter()
        response = await self.agent.run(case["question"])
        latency = time.perf_counter() - start_time
        
        ragas_metrics = await self.expert.score(case, response)
        judge_metrics = await self.judge.evaluate_multi_judge(case["question"], response, case.get("ground_truth", ""))
        
        return {
            "case": case,
            "response": response,
            "latency": latency,
            "ragas": ragas_metrics,
            "judge": judge_metrics
        }

# --- HELPERS ---

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)

def load_baseline(path="reports/baseline.json"):
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None

def export_markdown_report(v1_sum, v2_sum, approve, diffs):
    status = "✅ APPROVED" if approve else "❌ BLOCKED"
    m2 = v2_sum["metrics"]
    m1 = v1_sum["metrics"] if v1_sum else None
    
    # Chuẩn bị dữ liệu hiển thị (Tránh logic phức tạp trong f-string)
    score_v1 = f"{m1['avg_score']:.2f}" if m1 else "N/A"
    score_v2 = f"{m2['avg_score']:.2f}"
    score_delta = f"{m2['avg_score'] - m1['avg_score']:+.2f}" if m1 else "-"
    score_status = '🟢' if not m1 or m2['avg_score'] >= m1['avg_score']-0.1 else '🔴'

    hr_v1 = f"{m1['hit_rate']*100:.1f}%" if m1 else "N/A"
    hr_v2 = f"{m2['hit_rate']*100:.1f}%"
    hr_delta = f"{(m2['hit_rate'] - m1['hit_rate'])*100:+.1f}%" if m1 else "-"
    hr_status = '🟢' if not m1 or m2['hit_rate'] >= m1['hit_rate']-0.05 else '🔴'

    lat_v1 = f"{m1['avg_latency']:.2f}s" if m1 else "N/A"
    lat_v2 = f"{m2['avg_latency']:.2f}s"
    lat_delta = f"{m2['avg_latency'] - m1['avg_latency']:+.2f}s" if m1 else "-"
    lat_status = '🟢' if not m1 or m2['avg_latency'] <= m1['avg_latency']+0.5 else '🟡'

    md = f"""# 🚀 AI Agent Benchmark Report
**Status:** {status}
**Timestamp:** {v2_sum['metadata']['timestamp']}
**Version:** {v2_sum['metadata']['version']} vs {v1_sum['metadata']['version'] if v1_sum else 'None'}

## 📊 Performance Metrics
| Metric | Baseline (V1) | Current (V2) | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Avg Score** | {score_v1} | {score_v2} | {score_delta} | {score_status} |
| **Hit Rate** | {hr_v1} | {hr_v2} | {hr_delta} | {hr_status} |
| **Latency** | {lat_v1} | {lat_v2} | {lat_delta} | {lat_status} |

## 🔍 Regression Analysis
"""
    if not diffs:
        md += "\n✨ Không có case nào bị sụt giảm chất lượng."
    else:
        md += "| Câu hỏi | V1 Score | V2 Score | Lý do |\n| :--- | :--- | :--- | :--- |\n"
        for d in diffs[:10]:
            md += f"| {d['question'][:50]}... | {d['v1_score']} | {d['v2_score']} | {d['reasoning']} |\n"

    with open("reports/benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("📝 Báo cáo đã sẵn sàng tại: reports/benchmark_report.md")

# --- CORE LOGIC ---

async def run_benchmark_cycle(version_name):
    # Dữ liệu test mẫu
    dataset = [
        {"question": "VinUni nằm ở đâu?", "ground_truth": "Gia Lâm, Hà Nội"},
        {"question": "Làm thế nào để đăng ký học?", "ground_truth": "Qua cổng tuyển sinh trực tuyến"}
    ]
    
    runner = BenchmarkRunner(MockMainAgent(), ExpertEvaluator(), MultiModelJudge())
    results = await runner.run_all(dataset)
    
    total = len(results)
    summary = {
        "metadata": {
            "version": version_name, 
            "total": total, 
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        },
        "metrics": {
            "avg_score": sum(r["judge"]["final_score"] for r in results) / total,
            "avg_latency": sum(r["latency"] for r in results) / total,
            "hit_rate": sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / total,
            "agreement_rate": sum(r["judge"]["agreement_rate"] for r in results) / total
        },
        "raw_results": results
    }
    return results, summary

def regression_gate(v1_summary, v2_summary):
    m1, m2 = v1_summary["metrics"], v2_summary["metrics"]
    approve = True
    diffs = []
    
    # Kiểm tra ngưỡng (Thresholds)
    if (m2["avg_score"] - m1["avg_score"]) < -0.1: approve = False
    if (m2["hit_rate"] - m1["hit_rate"]) < -0.05: approve = False
    if (m2["avg_latency"] - m1["avg_latency"]) > 0.5: approve = False
    
    # So sánh từng case
    for r1, r2 in zip(v1_summary.get("raw_results", []), v2_summary.get("raw_results", [])):
        if r2["judge"]["final_score"] < r1["judge"]["final_score"]:
            diffs.append({
                "question": r1["case"]["question"],
                "v1_score": r1["judge"]["final_score"],
                "v2_score": r2["judge"]["final_score"],
                "reasoning": r2["judge"]["reasoning"]
            })
            
    return approve, diffs

async def main():
    CURRENT_VERSION = "Agent_V2_Optimized"
    
    v2_results, v2_summary = await run_benchmark_cycle(CURRENT_VERSION)
    v1_summary = load_baseline()
    
    if v1_summary:
        approve, diffs = regression_gate(v1_summary, v2_summary)
        export_markdown_report(v1_summary, v2_summary, approve, diffs)
        
        if approve:
            print(f"✅ APPROVE: Phiên bản {CURRENT_VERSION} đạt chuẩn.")
            save_json(v2_summary, "reports/baseline.json")
        else:
            print(f"❌ BLOCK: Phiên bản {CURRENT_VERSION} bị sụt giảm hiệu năng.")
    else:
        print("ℹ️ Khởi tạo Baseline lần đầu.")
        save_json(v2_summary, "reports/baseline.json")
        export_markdown_report(None, v2_summary, True, [])

    save_json(v2_results, f"reports/details_{CURRENT_VERSION}.json")

if __name__ == "__main__":
    asyncio.run(main())