import io
import json
import os
import statistics
import sys
import time
import warnings

warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
sys.path.insert(0, os.path.dirname(__file__))

from dotenv import load_dotenv

load_dotenv()

TESTSET_PATH = os.path.join(os.path.dirname(__file__), "testset.json")
RESULTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "results.json")

# ─── Import the ACTUAL production pipeline pieces from app.py ───────────────
print("Importing production pipeline from app.py (this loads all models)...")
import app as prod  # noqa: E402

_hybrid_retriever = prod._hybrid_retriever
_compressor = prod._compressor
question_answer_chain = prod.question_answer_chain

from langchain_groq import ChatGroq  # noqa: E402
from groq_deepeval_adapter import GroqDeepEvalLLM  # noqa: E402

from ragas import evaluate, EvaluationDataset  # noqa: E402
from ragas.llms import LangchainLLMWrapper  # noqa: E402
from ragas.embeddings import LangchainEmbeddingsWrapper  # noqa: E402
from ragas.metrics import (  # noqa: E402
    Faithfulness,
    AnswerRelevancy,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
)
from ragas.dataset_schema import SingleTurnSample  # noqa: E402

from deepeval.metrics import HallucinationMetric, BiasMetric, ToxicityMetric  # noqa: E402
from deepeval.test_case import LLMTestCase  # noqa: E402


def run_pipeline(question: str):
    """Drives the real retrieval -> rerank -> generation pipeline, timing each stage."""
    t0 = time.perf_counter()
    hybrid_docs = _hybrid_retriever.invoke(question)
    t1 = time.perf_counter()

    reranked_docs = _compressor.compress_documents(hybrid_docs, query=question)
    t2 = time.perf_counter()

    gen_result = question_answer_chain.invoke({
        "input": question, "context": reranked_docs, "chat_history": [],
    })
    answer = gen_result if isinstance(gen_result, str) else gen_result.get("answer", str(gen_result))
    t3 = time.perf_counter()

    return {
        "contexts": [d.page_content for d in reranked_docs],
        "answer": answer,
        "latency": {
            "retrieval_s": round(t1 - t0, 3),
            "rerank_s": round(t2 - t1, 3),
            "generation_s": round(t3 - t2, 3),
            "total_s": round(t3 - t0, 3),
        },
    }


def main():
    with open(TESTSET_PATH, "r", encoding="utf-8") as f:
        testset = json.load(f)
    print(f"Loaded {len(testset)} test questions.\n")

    print("Running production pipeline on every question (retrieval + rerank + generation)...")
    records = []
    for i, item in enumerate(testset, 1):
        q = item["question"]
        result = run_pipeline(q)
        records.append({**item, **result})
        print(f"[{i}/{len(testset)}] total={result['latency']['total_s']}s  Q: {q[:70]}")

    # ─── RAGAS ────────────────────────────────────────────────────────────
    print("\nScoring with RAGAS (faithfulness, answer_relevancy, context_precision, context_recall)...")
    judge_chat = ChatGroq(model="llama-3.3-70b-versatile", temperature=0, max_tokens=1024)
    ragas_llm = LangchainLLMWrapper(judge_chat)
    ragas_emb = LangchainEmbeddingsWrapper(prod.embeddings)

    samples = [
        SingleTurnSample(
            user_input=r["question"],
            retrieved_contexts=r["contexts"],
            response=r["answer"],
            reference=r["reference_answer"],
        )
        for r in records
    ]
    ds = EvaluationDataset(samples=samples)
    ragas_metrics = [
        Faithfulness(llm=ragas_llm),
        AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
        LLMContextPrecisionWithReference(llm=ragas_llm),
        LLMContextRecall(llm=ragas_llm),
    ]
    ragas_result = evaluate(dataset=ds, metrics=ragas_metrics, show_progress=True, raise_exceptions=False)
    ragas_df = ragas_result.to_pandas()
    for i, r in enumerate(records):
        r["ragas"] = {
            "faithfulness": _safe_float(ragas_df.loc[i, "faithfulness"]),
            "answer_relevancy": _safe_float(ragas_df.loc[i, "answer_relevancy"]),
            "context_precision": _safe_float(ragas_df.loc[i, "llm_context_precision_with_reference"]),
            "context_recall": _safe_float(ragas_df.loc[i, "context_recall"]),
        }

    # ─── DeepEval ─────────────────────────────────────────────────────────
    print("\nScoring with DeepEval (hallucination, bias, toxicity)...")
    deepeval_judge = GroqDeepEvalLLM(judge_chat, "groq-llama-3.3-70b-versatile")
    hallucination_metric = HallucinationMetric(model=deepeval_judge, threshold=0.5, include_reason=False, async_mode=False)
    bias_metric = BiasMetric(model=deepeval_judge, threshold=0.5, include_reason=False, async_mode=False)
    toxicity_metric = ToxicityMetric(model=deepeval_judge, threshold=0.5, include_reason=False, async_mode=False)

    for i, r in enumerate(records, 1):
        tc = LLMTestCase(
            input=r["question"],
            actual_output=r["answer"],
            context=r["contexts"],
        )
        de_scores = {}
        for name, metric in [("hallucination", hallucination_metric), ("bias", bias_metric), ("toxicity", toxicity_metric)]:
            try:
                metric.measure(tc)
                de_scores[name] = round(float(metric.score), 4)
            except Exception as e:
                de_scores[name] = None
                print(f"  [{i}] {name} failed: {e}")
        r["deepeval"] = de_scores
        print(f"[{i}/{len(records)}] hallucination={de_scores.get('hallucination')} bias={de_scores.get('bias')} toxicity={de_scores.get('toxicity')}")

    # ─── Aggregate + save ─────────────────────────────────────────────────
    with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(records, f, indent=2, ensure_ascii=False)
    print(f"\nSaved full results to {RESULTS_JSON_PATH}")

    print_summary(records)


def _safe_float(v):
    try:
        f = float(v)
        return None if f != f else round(f, 4)  # NaN check
    except Exception:
        return None


def _agg(records, path):
    vals = []
    for r in records:
        d = r
        for k in path:
            d = d.get(k, {}) if isinstance(d, dict) else {}
        if isinstance(d, (int, float)):
            vals.append(d)
    return vals


def print_summary(records):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        vals = [r["ragas"][metric] for r in records if r["ragas"].get(metric) is not None]
        if vals:
            print(f"RAGAS {metric:22s}: mean={statistics.mean(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}  n={len(vals)}")

    for metric in ["hallucination", "bias", "toxicity"]:
        vals = [r["deepeval"][metric] for r in records if r["deepeval"].get(metric) is not None]
        if vals:
            print(f"DeepEval {metric:20s}: mean={statistics.mean(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}  n={len(vals)}")

    for stage in ["retrieval_s", "rerank_s", "generation_s", "total_s"]:
        vals = sorted(r["latency"][stage] for r in records)
        n = len(vals)
        p50 = vals[n // 2]
        p95 = vals[min(n - 1, int(n * 0.95))]
        print(f"Latency {stage:15s}: mean={statistics.mean(vals):.3f}s  p50={p50:.3f}s  p95={p95:.3f}s  max={max(vals):.3f}s")


if __name__ == "__main__":
    main()
