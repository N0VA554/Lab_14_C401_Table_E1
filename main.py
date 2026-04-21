import asyncio
import json
import os
import time
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Optional

from agent.main_agent import MainAgent
from engine.llm_judge import LLMJudge
from engine.retrieval_eval import RetrievalEvaluator


# ---------------------------------------------------------------------------
# Expert Evaluator — wraps RetrievalEvaluator + computes RAGAS-style metrics
# ---------------------------------------------------------------------------

class ExpertEvaluator:
    """
    Computes retrieval metrics (Hit Rate, MRR) by matching retrieved chunk IDs
    against expected chunk IDs derived from each test case's context field.
    """

    def __init__(self, chunks: List[Dict]):
        self.chunks = chunks
        self.retrieval_eval = RetrievalEvaluator()

    def _find_expected_chunks(self, context: str) -> List[str]:
        """Find top-2 chunk IDs whose text best matches the test case context."""
        if not context:
            return []
        ctx_lower = context.lower()
        ctx_words = set(ctx_lower.split())
        scored = []
        for chunk in self.chunks:
            chunk_words = set(chunk["text"].lower().split())
            score = len(ctx_words & chunk_words)
            scored.append((score, chunk["chunk_id"]))
        scored.sort(reverse=True)
        return [cid for score, cid in scored[:2] if score > 0]

    async def score(self, test_case: Dict, response: Dict) -> Dict:
        context = test_case.get("context", "")
        expected_ids = self._find_expected_chunks(context)
        retrieved_ids = response.get("retrieved_ids", [])

        hit_rate = self.retrieval_eval.calculate_hit_rate(expected_ids, retrieved_ids)
        mrr = self.retrieval_eval.calculate_mrr(expected_ids, retrieved_ids)

        return {
            "faithfulness": 1.0 if hit_rate > 0 else 0.5,
            "relevancy": 0.9 if mrr > 0 else 0.6,
            "retrieval": {
                "hit_rate": hit_rate,
                "mrr": mrr,
                "expected_ids": expected_ids,
                "retrieved_ids": retrieved_ids,
            },
        }


# ---------------------------------------------------------------------------
# Async Benchmark Runner
# ---------------------------------------------------------------------------

class BenchmarkRunner:
    def __init__(self, agent: MainAgent, evaluator: ExpertEvaluator, judge: LLMJudge):
        self.agent = agent
        self.evaluator = evaluator
        self.judge = judge

    async def _run_single(self, test_case: Dict) -> Dict:
        start = time.perf_counter()
        response = await self.agent.query(test_case["question"])
        latency = time.perf_counter() - start

        ragas = await self.evaluator.score(test_case, response)
        judge_result = await self.judge.evaluate_multi_judge(
            test_case["question"],
            response["answer"],
            test_case.get("expected_answer", ""),
        )

        return {
            "test_case": test_case["question"],
            "difficulty": test_case.get("difficulty", "Unknown"),
            "agent_response": response["answer"],
            "latency": round(latency, 3),
            "cost_usd": response["metadata"].get("cost_usd", 0),
            "tokens_used": response["metadata"].get("tokens_used", 0),
            "ragas": ragas,
            "judge": judge_result,
            "status": "pass" if judge_result["final_score"] >= 3 else "fail",
        }

    async def run_all(self, dataset: List[Dict], batch_size: int = 5) -> List[Dict]:
        results = []
        total = len(dataset)
        for i in range(0, total, batch_size):
            batch = dataset[i : i + batch_size]
            print(f"  📦 Batch {i//batch_size + 1}/{(total-1)//batch_size + 1} ({len(batch)} cases)...")
            batch_results = await asyncio.gather(*[self._run_single(c) for c in batch])
            results.extend(batch_results)
        return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_chunks() -> List[Dict]:
    path = Path("data/chunks.jsonl")
    if not path.exists():
        return []
    chunks = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                chunks.append(json.loads(line))
    return chunks


def load_dataset() -> List[Dict]:
    path = Path("data/golden_set.jsonl")
    if not path.exists():
        print("❌ data/golden_set.jsonl not found.")
        return []
    dataset = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                dataset.append(json.loads(line))
    return dataset


def compute_summary(version: str, results: List[Dict]) -> Dict:
    n = len(results)
    if n == 0:
        return {}
    pass_count = sum(1 for r in results if r["status"] == "pass")
    total_cost = sum(r.get("cost_usd", 0) for r in results)
    total_tokens = sum(r.get("tokens_used", 0) for r in results)

    return {
        "metadata": {
            "version": version,
            "total": n,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "cost_usd": round(total_cost, 4),
            "total_tokens": total_tokens,
        },
        "metrics": {
            "avg_score": round(sum(r["judge"]["final_score"] for r in results) / n, 3),
            "pass_rate": round(pass_count / n, 3),
            "hit_rate": round(sum(r["ragas"]["retrieval"]["hit_rate"] for r in results) / n, 3),
            "avg_mrr": round(sum(r["ragas"]["retrieval"]["mrr"] for r in results) / n, 3),
            "avg_latency": round(sum(r["latency"] for r in results) / n, 3),
            "agreement_rate": round(
                sum(r["judge"]["agreement_rate"] for r in results) / n, 3
            ),
        },
        "raw_results": results,
    }


def save_json(data, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2)


def regression_gate(v1: Dict, v2: Dict):
    """Auto Release Gate: approve if V2 doesn't regress on key metrics."""
    m1, m2 = v1["metrics"], v2["metrics"]
    reasons = []
    approve = True

    if m2["avg_score"] - m1["avg_score"] < -0.1:
        approve = False
        reasons.append(f"Score regressed: {m1['avg_score']} → {m2['avg_score']}")
    if m2["hit_rate"] - m1["hit_rate"] < -0.05:
        approve = False
        reasons.append(f"Hit rate regressed: {m1['hit_rate']} → {m2['hit_rate']}")
    if m2["avg_latency"] - m1["avg_latency"] > 1.0:
        approve = False
        reasons.append(f"Latency increased: {m1['avg_latency']}s → {m2['avg_latency']}s")

    diffs = []
    for r1, r2 in zip(v1.get("raw_results", []), v2.get("raw_results", [])):
        if r2["judge"]["final_score"] < r1["judge"]["final_score"]:
            diffs.append({
                "question": r1["test_case"],
                "v1_score": r1["judge"]["final_score"],
                "v2_score": r2["judge"]["final_score"],
                "reasoning": r2["judge"].get("reasoning", ""),
            })

    return approve, reasons, diffs


def export_markdown_report(v1: Optional[Dict], v2: Dict, approve: bool, reasons: List, diffs: List):
    status = "✅ APPROVED" if approve else "❌ BLOCKED"
    m2 = v2["metrics"]
    m1 = v1["metrics"] if v1 else None

    def fmt(val, prev, fmt_fn, better_if="higher"):
        cur = fmt_fn(val)
        if prev is None:
            return cur, "-", "🟡"
        delta = val - prev
        sign = "+" if delta >= 0 else ""
        is_better = delta >= 0 if better_if == "higher" else delta <= 0
        icon = "🟢" if is_better else "🔴"
        return cur, f"{sign}{fmt_fn(delta)}", icon

    score_cur, score_d, score_ic = fmt(m2["avg_score"], m1["avg_score"] if m1 else None, lambda x: f"{x:.3f}")
    hr_cur, hr_d, hr_ic = fmt(m2["hit_rate"], m1["hit_rate"] if m1 else None, lambda x: f"{x:.3f}")
    mrr_cur, mrr_d, mrr_ic = fmt(m2["avg_mrr"], m1["avg_mrr"] if m1 else None, lambda x: f"{x:.3f}")
    lat_cur, lat_d, lat_ic = fmt(m2["avg_latency"], m1["avg_latency"] if m1 else None, lambda x: f"{x:.3f}s", "lower")
    pr_cur, pr_d, pr_ic = fmt(m2["pass_rate"], m1["pass_rate"] if m1 else None, lambda x: f"{x:.1%}")
    agr_cur, agr_d, agr_ic = fmt(m2["agreement_rate"], m1["agreement_rate"] if m1 else None, lambda x: f"{x:.3f}")

    score_v1 = f"{m1['avg_score']:.3f}" if m1 else "N/A"
    hr_v1 = f"{m1['hit_rate']:.3f}" if m1 else "N/A"
    mrr_v1 = f"{m1['avg_mrr']:.3f}" if m1 else "N/A"
    lat_v1 = f"{m1['avg_latency']:.3f}s" if m1 else "N/A"
    pr_v1 = f"{m1['pass_rate']:.1%}" if m1 else "N/A"
    agr_v1 = f"{m1['agreement_rate']:.3f}" if m1 else "N/A"

    v1_name = v1["metadata"]["version"] if v1 else "None"

    md = f"""# 🚀 AI Agent Benchmark Report
**Decision:** {status}
**Timestamp:** {v2['metadata']['timestamp']}
**Versions:** {v1_name} → {v2['metadata']['version']}
**Total cases:** {v2['metadata']['total']} | **Cost:** ${v2['metadata']['cost_usd']:.4f} | **Tokens:** {v2['metadata']['total_tokens']:,}

## 📊 Performance Metrics
| Metric | Baseline ({v1_name}) | Current ({v2['metadata']['version']}) | Delta | Status |
| :--- | :--- | :--- | :--- | :--- |
| **Avg Judge Score** | {score_v1} | {score_cur} | {score_d} | {score_ic} |
| **Pass Rate** | {pr_v1} | {pr_cur} | {pr_d} | {pr_ic} |
| **Hit Rate** | {hr_v1} | {hr_cur} | {hr_d} | {hr_ic} |
| **MRR** | {mrr_v1} | {mrr_cur} | {mrr_d} | {mrr_ic} |
| **Avg Latency** | {lat_v1} | {lat_cur} | {lat_d} | {lat_ic} |
| **Agreement Rate** | {agr_v1} | {agr_cur} | {agr_d} | {agr_ic} |

## 🔍 Regression Analysis
"""
    if reasons:
        md += "**Block reasons:**\n"
        for r in reasons:
            md += f"- {r}\n"
        md += "\n"

    if not diffs:
        md += "✨ Không có case nào bị sụt giảm chất lượng.\n"
    else:
        md += f"**{len(diffs)} case(s) regressed:**\n\n"
        md += "| Câu hỏi | V1 | V2 | Lý do |\n| :--- | :--- | :--- | :--- |\n"
        for d in diffs[:10]:
            q = d["question"][:55] + "..." if len(d["question"]) > 55 else d["question"]
            md += f"| {q} | {d['v1_score']} | {d['v2_score']} | {d['reasoning'][:60]} |\n"

    os.makedirs("reports", exist_ok=True)
    with open("reports/benchmark_report.md", "w", encoding="utf-8") as f:
        f.write(md)
    print("📝 reports/benchmark_report.md saved.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

async def run_benchmark(version_name: str, top_k: int, dataset: List[Dict], chunks: List[Dict]) -> Dict:
    print(f"\n🚀 Running benchmark: {version_name} (top_k={top_k})...")
    agent = MainAgent(top_k=top_k)
    evaluator = ExpertEvaluator(chunks)
    judge = LLMJudge()
    runner = BenchmarkRunner(agent, evaluator, judge)
    results = await runner.run_all(dataset, batch_size=5)
    summary = compute_summary(version_name, results)
    print(f"  ✅ Done. Avg score: {summary['metrics']['avg_score']:.3f} | Hit rate: {summary['metrics']['hit_rate']:.3f}")
    return summary


async def main():
    dataset = load_dataset()
    if not dataset:
        return
    print(f"📚 Loaded {len(dataset)} test cases from golden_set.jsonl")

    chunks = load_chunks()
    print(f"🗂️  Loaded {len(chunks)} chunks")

    # V1 baseline: top_k=3
    v1_summary = await run_benchmark("Agent_V1_Base", top_k=3, dataset=dataset, chunks=chunks)

    # V2 optimized: top_k=5 (more context = better recall)
    v2_summary = await run_benchmark("Agent_V2_Optimized", top_k=5, dataset=dataset, chunks=chunks)

    # Regression gate
    approve, reasons, diffs = regression_gate(v1_summary, v2_summary)
    decision = "✅ APPROVE" if approve else "❌ BLOCK RELEASE"
    print(f"\n📊 Regression Gate: {decision}")
    print(f"   V1 avg_score={v1_summary['metrics']['avg_score']:.3f}  V2 avg_score={v2_summary['metrics']['avg_score']:.3f}")
    if reasons:
        for r in reasons:
            print(f"   ⛔ {r}")

    # Save reports
    os.makedirs("reports", exist_ok=True)

    # summary.json = V2 summary + comparison vs V1
    summary_out = {
        **v2_summary,
        "regression": {
            "baseline_version": v1_summary["metadata"]["version"],
            "approve": approve,
            "block_reasons": reasons,
            "regressed_cases": len(diffs),
            "v1_metrics": v1_summary["metrics"],
            "v2_metrics": v2_summary["metrics"],
        },
    }
    # Remove raw_results from summary.json (kept in benchmark_results.json)
    summary_out_clean = {k: v for k, v in summary_out.items() if k != "raw_results"}
    save_json(summary_out_clean, "reports/summary.json")

    save_json(v2_summary["raw_results"], "reports/benchmark_results.json")

    export_markdown_report(v1_summary, v2_summary, approve, reasons, diffs)

    print("\n📁 Reports saved:")
    print("   reports/summary.json")
    print("   reports/benchmark_results.json")
    print("   reports/benchmark_report.md")

    # Cost report
    total_cost = v1_summary["metadata"]["cost_usd"] + v2_summary["metadata"]["cost_usd"]
    total_tokens = v1_summary["metadata"]["total_tokens"] + v2_summary["metadata"]["total_tokens"]
    print(f"\n💰 Cost report:")
    print(f"   V1: ${v1_summary['metadata']['cost_usd']:.4f} | {v1_summary['metadata']['total_tokens']:,} tokens")
    print(f"   V2: ${v2_summary['metadata']['cost_usd']:.4f} | {v2_summary['metadata']['total_tokens']:,} tokens")
    print(f"   Total: ${total_cost:.4f} | {total_tokens:,} tokens")
    print(f"   Cost per eval: ${total_cost / (len(dataset)*2):.6f}")


if __name__ == "__main__":
    asyncio.run(main())
