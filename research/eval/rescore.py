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

RESULTS_JSON_PATH = os.path.join(os.path.dirname(__file__), "results.json")
JUDGE_MODEL = "llama-3.1-8b-instant"

from langchain_groq import ChatGroq  # noqa: E402
from groq_deepeval_adapter import GroqDeepEvalLLM  # noqa: E402

from src.helper import download_hugging_face_embeddings  # noqa: E402

from ragas import evaluate, EvaluationDataset, RunConfig  # noqa: E402
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


def measure_with_retry(metric, tc, max_retries=5, base_wait=15):
    for attempt in range(max_retries):
        try:
            metric.measure(tc)
            return round(float(metric.score), 4)
        except Exception as e:
            msg = str(e)
            if "429" in msg or "rate_limit" in msg.lower():
                wait = base_wait * (attempt + 1)
                print(f"    rate limited, waiting {wait}s (attempt {attempt+1}/{max_retries})...")
                time.sleep(wait)
                continue
            print(f"    failed (non-rate-limit): {e}")
            return None
    print("    giving up after retries")
    return None


def main():
    with open(RESULTS_JSON_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"Loaded {len(records)} records with cached pipeline outputs (contexts/answer/latency).")

    judge_chat = ChatGroq(model=JUDGE_MODEL, temperature=0, max_tokens=1024)

    missing_ragas = [
        r for r in records
        if not r.get("ragas") or any(r["ragas"].get(k) is None for k in
            ("faithfulness", "answer_relevancy", "context_precision", "context_recall"))
    ]
    print(f"{len(missing_ragas)} records need (re)scoring with RAGAS.")

    if missing_ragas:
        ragas_llm = LangchainLLMWrapper(judge_chat)
        ragas_emb = LangchainEmbeddingsWrapper(download_hugging_face_embeddings())
        samples = [
            SingleTurnSample(
                user_input=r["question"],
                retrieved_contexts=r["contexts"],
                response=r["answer"],
                reference=r["reference_answer"],
            )
            for r in missing_ragas
        ]
        ds = EvaluationDataset(samples=samples)
        ragas_metrics = [
            Faithfulness(llm=ragas_llm),
            AnswerRelevancy(llm=ragas_llm, embeddings=ragas_emb),
            LLMContextPrecisionWithReference(llm=ragas_llm),
            LLMContextRecall(llm=ragas_llm),
        ]
        run_config = RunConfig(max_workers=2, timeout=180, max_retries=8, max_wait=60)
        ragas_result = evaluate(
            dataset=ds, metrics=ragas_metrics, show_progress=True,
            raise_exceptions=False, run_config=run_config,
        )
        ragas_df = ragas_result.to_pandas()
        for i, r in enumerate(missing_ragas):
            r["ragas"] = {
                "faithfulness": _safe_float(ragas_df.loc[i, "faithfulness"]),
                "answer_relevancy": _safe_float(ragas_df.loc[i, "answer_relevancy"]),
                "context_precision": _safe_float(ragas_df.loc[i, "llm_context_precision_with_reference"]),
                "context_recall": _safe_float(ragas_df.loc[i, "context_recall"]),
            }
        with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
            json.dump(records, f, indent=2, ensure_ascii=False)
        print("Saved RAGAS rescoring progress.")

    missing_deepeval = [
        r for r in records
        if not r.get("deepeval") or any(r["deepeval"].get(k) is None for k in
            ("hallucination", "bias", "toxicity"))
    ]
    print(f"\n{len(missing_deepeval)} records need (re)scoring with DeepEval.")

    if missing_deepeval:
        deepeval_judge = GroqDeepEvalLLM(judge_chat, f"groq-{JUDGE_MODEL}")
        hallucination_metric = HallucinationMetric(model=deepeval_judge, threshold=0.5, include_reason=False, async_mode=False)
        bias_metric = BiasMetric(model=deepeval_judge, threshold=0.5, include_reason=False, async_mode=False)
        toxicity_metric = ToxicityMetric(model=deepeval_judge, threshold=0.5, include_reason=False, async_mode=False)

        for i, r in enumerate(missing_deepeval, 1):
            tc = LLMTestCase(input=r["question"], actual_output=r["answer"], context=r["contexts"])
            de_scores = r.get("deepeval") or {}
            for name, metric in [("hallucination", hallucination_metric), ("bias", bias_metric), ("toxicity", toxicity_metric)]:
                if de_scores.get(name) is not None:
                    continue
                de_scores[name] = measure_with_retry(metric, tc)
                time.sleep(2)  # small pacing gap between calls
            r["deepeval"] = de_scores
            print(f"[{i}/{len(missing_deepeval)}] hallucination={de_scores.get('hallucination')} bias={de_scores.get('bias')} toxicity={de_scores.get('toxicity')}")

            with open(RESULTS_JSON_PATH, "w", encoding="utf-8") as f:
                json.dump(records, f, indent=2, ensure_ascii=False)

    print(f"\nSaved full results to {RESULTS_JSON_PATH}")
    print_summary(records)


def _safe_float(v):
    try:
        f = float(v)
        return None if f != f else round(f, 4)
    except Exception:
        return None


def print_summary(records):
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    for metric in ["faithfulness", "answer_relevancy", "context_precision", "context_recall"]:
        vals = [r["ragas"][metric] for r in records if r.get("ragas", {}).get(metric) is not None]
        if vals:
            print(f"RAGAS {metric:22s}: mean={statistics.mean(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}  n={len(vals)}/{len(records)}")
        else:
            print(f"RAGAS {metric:22s}: NO DATA")

    for metric in ["hallucination", "bias", "toxicity"]:
        vals = [r["deepeval"][metric] for r in records if r.get("deepeval", {}).get(metric) is not None]
        if vals:
            print(f"DeepEval {metric:20s}: mean={statistics.mean(vals):.3f}  min={min(vals):.3f}  max={max(vals):.3f}  n={len(vals)}/{len(records)}")
        else:
            print(f"DeepEval {metric:20s}: NO DATA")

    for stage in ["retrieval_s", "rerank_s", "generation_s", "total_s"]:
        vals = sorted(r["latency"][stage] for r in records)
        n = len(vals)
        p50 = vals[n // 2]
        p95 = vals[min(n - 1, int(n * 0.95))]
        print(f"Latency {stage:15s}: mean={statistics.mean(vals):.3f}s  p50={p50:.3f}s  p95={p95:.3f}s  max={max(vals):.3f}s")


if __name__ == "__main__":
    main()
