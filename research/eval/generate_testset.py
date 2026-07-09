import io
import json
import os
import random
import re
import sys
import warnings

warnings.filterwarnings("ignore")

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

from dotenv import load_dotenv

load_dotenv()

from qdrant_client import QdrantClient
from langchain_qdrant import QdrantVectorStore
from langchain_groq import ChatGroq
from src.helper import download_hugging_face_embeddings

COLLECTION = "medical_chatbot_base"
N_QUESTIONS = 25
OUT_PATH = os.path.join(os.path.dirname(__file__), "testset.json")

GEN_PROMPT = """You are creating a test question for evaluating a medical RAG chatbot.
Given the medical text below, write ONE realistic question a patient might ask that this text directly and fully answers, plus a concise 2-3 sentence reference answer based ONLY on this text.

Medical text:
\"\"\"{chunk}\"\"\"

Respond in strict JSON only, no markdown fences, no extra commentary:
{{"question": "...", "reference_answer": "..."}}"""


def main():
    print("Loading embedding model...")
    embeddings = download_hugging_face_embeddings()
    client = QdrantClient(host="localhost", port=6333)
    store = QdrantVectorStore(client=client, collection_name=COLLECTION, embedding=embeddings)

    total = client.count(collection_name=COLLECTION, exact=True).count
    print(f"Collection has {total} chunks")

    rng = random.Random(42)
    sample_size = N_QUESTIONS * 4
    target_indices = set(rng.sample(range(total), min(sample_size, total)))

    picked_docs = []
    scroll_offset = None
    i = 0
    while True:
        results, scroll_offset = client.scroll(
            collection_name=COLLECTION, limit=500, offset=scroll_offset,
            with_payload=True, with_vectors=False,
        )
        for point in results:
            if i in target_indices:
                text = point.payload.get("page_content") or point.payload.get("document", "")
                meta = point.payload.get("metadata", {})
                if text and len(text.strip()) > 300:
                    picked_docs.append({"id": point.id, "text": text, "metadata": meta})
            i += 1
        if scroll_offset is None:
            break

    rng.shuffle(picked_docs)
    print(f"Sampled {len(picked_docs)} candidate chunks, generating up to {N_QUESTIONS} Q&A pairs...")

    judge = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.4, max_tokens=400)

    testset = []
    for doc in picked_docs:
        if len(testset) >= N_QUESTIONS:
            break
        try:
            resp = judge.invoke(GEN_PROMPT.format(chunk=doc["text"][:1500])).content.strip()
            resp = re.sub(r"^```(json)?|```$", "", resp, flags=re.MULTILINE).strip()
            data = json.loads(resp)
            q, ref = data["question"].strip(), data["reference_answer"].strip()
            if not q or not ref:
                continue
        except Exception as e:
            print(f"  skip (generation failed): {e}")
            continue

        hits = store.similarity_search(q, k=8)
        verified = any(
            doc["text"][:200] in h.page_content or h.page_content[:200] in doc["text"]
            for h in hits
        )

        testset.append({
            "question": q,
            "reference_answer": ref,
            "source_chunk_id": str(doc["id"]),
            "source_metadata": doc["metadata"],
            "verified_retrievable": verified,
        })
        print(f"[{len(testset)}/{N_QUESTIONS}] verified={verified}  Q: {q[:80]}")

    client.close()

    n_verified = sum(t["verified_retrievable"] for t in testset)
    print(f"\n{n_verified}/{len(testset)} questions verified retrievable by the actual pipeline.")

    unverified = [t for t in testset if not t["verified_retrievable"]]
    if unverified:
        print(f"WARNING: {len(unverified)} question(s) NOT confirmed retrievable — kept in set but flagged:")
        for t in unverified:
            print(f"  - {t['question'][:90]}")

    os.makedirs(os.path.dirname(OUT_PATH), exist_ok=True)
    with open(OUT_PATH, "w", encoding="utf-8") as f:
        json.dump(testset, f, indent=2, ensure_ascii=False)
    print(f"Saved {len(testset)} Q&A pairs to {OUT_PATH}")


if __name__ == "__main__":
    main()
