#!/usr/bin/env python3
"""
Multi-stage RAG query (Retrieve -> Rerank -> Diversify -> Generate) for Synapse

This module consumes artifacts produced by prior steps:
  - artifacts/embeddings.jsonl (summary_text, full_text, embedding_summary, embedding_full, metadata)

Retrieval pipeline:
  1) Dense multi-vector recall: FAISS IP search over summary and full vectors
  2) Sparse recall: BM25 over full_text
  3) Fusion: Reciprocal Rank Fusion (RRF), with per-document caps
  4) Rerank: Flashrank or MiniLM CrossEncoder, local
  5) Diversify: MMR to reduce redundancy
  6) Generate: Gemini 2.5 Pro via LiteLLM proxy with strict inline citations [n]

Environment variables (LiteLLM proxy):
  - LITELLM_API_KEY: API key for your proxy
  - LITELLM_BASE_URL: Base URL for the proxy (e.g., http://localhost:4000)
  - LITELLM_MODEL (optional): Default "gemini/gemini-2.5-pro"

Usage:
  python pipeline/query.py --question "What were the timing constraints for the version 2 Tensix L2 cache controller?"
"""

from __future__ import annotations

import argparse
import json
import os
import re
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np


# Optional rerankers
FLASHRANK_OK = False
try:  # pragma: no cover - optional dependency
    from flashrank import Ranker, RerankRequest  # type: ignore

    FLASHRANK_OK = True
except Exception:
    FLASHRANK_OK = False

CE = None
try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder  # type: ignore

    CE = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
except Exception:
    CE = None


def _norm_rows(A: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(A, axis=1, keepdims=True) + 1e-9
    return (A / n).astype("float32")


def read_jsonl(path: str) -> Iterable[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def load_corpus(embeddings_path: str) -> List[Dict[str, object]]:
    items: List[Dict[str, object]] = []
    for rec in read_jsonl(embeddings_path):
        emb_s = rec.get("embedding_summary")
        emb_f = rec.get("embedding_full")
        if not emb_s or not emb_f:
            # Skip malformed rows
            continue
        items.append(
            {
                "id": rec.get("id"),
                "document_id": rec.get("document_id"),
                "source_path": rec.get("source_path"),
                "source_type": rec.get("source_type"),
                "metadata": rec.get("metadata", {}),
                "summary_text": rec.get("summary_text", ""),
                "full_text": rec.get("full_text", ""),
                "emb_sum": np.asarray(rec.get("embedding_summary"), dtype="float32"),
                "emb_full": np.asarray(rec.get("embedding_full"), dtype="float32"),
            }
        )
    return items


def build_indices(items: List[Dict[str, object]]):
    import faiss  # type: ignore
    from rank_bm25 import BM25Okapi  # type: ignore

    E_sum = _norm_rows(np.vstack([it["emb_sum"] for it in items]))
    E_full = _norm_rows(np.vstack([it["emb_full"] for it in items]))
    idx_sum = faiss.IndexFlatIP(E_sum.shape[1])
    idx_full = faiss.IndexFlatIP(E_full.shape[1])
    idx_sum.add(E_sum)
    idx_full.add(E_full)
    bm25 = BM25Okapi([str(it["full_text"]).lower().split() for it in items])
    return idx_sum, idx_full, E_sum, E_full, bm25


def load_query_encoder(default_model: str = "BAAI/bge-large-en-v1.5"):
    # Use the same retrieval model used during embedding, if known. Allow override via env.
    model_name = os.environ.get("EMBED_MODEL", default_model)
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore

        model = SentenceTransformer(model_name)
        return model_name, model
    except Exception as exc:
        raise RuntimeError(
            "sentence-transformers is required to encode queries for dense retrieval."
        ) from exc


def encode_query(q: str, model) -> np.ndarray:
    # bge/e5 families expect "query: " prefix, while docs were embedded with "passage: "
    text = f"query: {q}"
    vec = model.encode([text], normalize_embeddings=True)
    return np.asarray(vec, dtype="float32")


def dense_search(qv: np.ndarray, faiss_index, k: int) -> List[int]:
    D, I = faiss_index.search(qv, k)
    return I[0].tolist()


def sparse_search(query: str, bm25, k: int) -> List[int]:
    toks = query.lower().split()
    scores = bm25.get_scores(toks)
    order = np.argsort(scores)[::-1][:k]
    return order.tolist()


def rrf(rank_lists: List[List[int]], weights: List[float], rrf_k: int = 60, base: int = 60) -> List[int]:
    scores: Dict[int, float] = {}
    for lst, w in zip(rank_lists, weights):
        for r, idx in enumerate(lst[:rrf_k], start=1):
            scores[idx] = scores.get(idx, 0.0) + w * (1.0 / (base + r))
    return [i for i, _ in sorted(scores.items(), key=lambda x: x[1], reverse=True)]


def per_doc_cap(indices: List[int], items: List[Dict[str, object]], cap: int) -> List[int]:
    counts: Dict[str, int] = {}
    out: List[int] = []
    for i in indices:
        d = str(items[i]["document_id"])
        if counts.get(d, 0) < cap:
            out.append(i)
            counts[d] = counts.get(d, 0) + 1
    return out


def rerank(query: str, candidate_indices: List[int], items: List[Dict[str, object]], topn: int = 20) -> List[int]:
    if not candidate_indices:
        return []
    passages = [
        {"id": str(i), "text": str(items[i]["full_text"])[:2000]} for i in candidate_indices
    ]
    if FLASHRANK_OK:
        ranker = Ranker()
        req = RerankRequest(query=query, passages=passages)
        ranked = ranker.rerank(req)
        # Handle both dict and object returns from flashrank
        result = []
        for p in ranked[:topn]:
            if hasattr(p, 'id'):
                result.append(int(p.id))
            elif isinstance(p, dict):
                result.append(int(p['id']))
            else:
                result.append(int(str(p)))
        return result
    if CE is not None:
        pairs = [(query, str(items[i]["full_text"])) for i in candidate_indices]
        scores = CE.predict(pairs)
        order = np.argsort(scores)[::-1][:topn]
        return [candidate_indices[i] for i in order]
    return candidate_indices[:topn]


def mmr_select(query_vec: np.ndarray, candidate_indices: List[int], E_full: np.ndarray, lambda_: float, final_k: int) -> List[int]:
    selected: List[int] = []
    if not candidate_indices:
        return selected
    qv = query_vec.reshape(-1)
    cand = candidate_indices.copy()
    sim_q = (E_full[cand] @ qv)
    while cand and len(selected) < final_k:
        if not selected:
            j = int(np.argmax(sim_q))
            selected.append(cand[j])
            cand.pop(j)
            sim_q = np.delete(sim_q, j)
            continue
        cur = np.asarray([E_full[i] for i in selected])
        # Diversity term: max similarity to items already selected
        div = []
        for idx in cand:
            div.append(float(np.max(cur @ E_full[idx])))
        div = np.asarray(div, dtype=np.float32)
        mmr = lambda_ * sim_q + (1.0 - lambda_) * (-div)
        j = int(np.argmax(mmr))
        selected.append(cand[j])
        cand.pop(j)
        sim_q = np.delete(sim_q, j)
    return selected


def format_citations(indices: List[int], items: List[Dict[str, object]]) -> Tuple[str, Dict[int, str]]:
    mapping: Dict[int, str] = {}
    lines: List[str] = []
    for n, i in enumerate(indices, start=1):
        it = items[i]
        src = str(it.get("source_path"))
        meta = dict(it.get("metadata", {}))
        page = meta.get("page_number") or meta.get("slide_number")
        mapping[i] = f"[{n}]"
        label = f"{src}" + (f" (page {int(page)})" if page else "")
        lines.append(f"[{n}] {label}")
    return "\n".join(lines), mapping


def build_prompt(question: str, final_indices: List[int], items: List[Dict[str, object]]) -> Tuple[str, str]:
    ctx_blocks: List[str] = []
    for n, i in enumerate(final_indices, start=1):
        text = str(items[i]["full_text"]).strip()
        if len(text) > 2000:
            text = text[:2000]
        ctx_blocks.append(f"[{n}] {text}")
    context = "\n\n".join(ctx_blocks)
    system = (
        "You are a precise engineering assistant. Answer ONLY using the provided context. "
        "Cite sources inline like [1], [2] immediately after the claims they support. "
        "If the answer is not found, say you don't know."
    )
    user = f"Question: {question}\n\nContext:\n{context}\n\nProvide a concise answer with inline citations and a Sources list."
    return system, user


def call_gemini_via_litellm(system: str, user: str, timeout: int = 60) -> str:
    try:
        from litellm import completion  # type: ignore
    except Exception as exc:
        raise RuntimeError("Please install 'litellm' to use the LiteLLM proxy (pip install litellm)") from exc

    api_key = os.getenv("LITELLM_API_KEY")
    base_url = os.getenv("LITELLM_BASE_URL")
    if not api_key or not base_url:
        raise RuntimeError("LITELLM_API_KEY and LITELLM_BASE_URL must be set in the environment.")

    model = os.getenv("LITELLM_MODEL", "gemini/gemini-2.5-pro")
    resp = completion(
        model=model,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        api_key=api_key,
        base_url=base_url,
        custom_llm_provider="openai",  # Force OpenAI-compatible routing
        timeout=timeout,
    )
    # Extract OpenAI-style content
    choice = resp.choices[0]
    message = getattr(choice, "message", None) or choice.get("message")
    content = getattr(message, "content", None) or (message.get("content") if isinstance(message, dict) else None)
    return str(content) if content is not None else str(resp)


def answer(
    question: str,
    embeddings_path: str,
    k_dense_sum: int = 60,
    k_dense_full: int = 60,
    k_sparse: int = 60,
    per_doc: int = 4,
    final_k: int = 8,
    lambda_mmr: float = 0.7,
    timeout: int = 60,
) -> Tuple[str, str]:
    items = load_corpus(embeddings_path)
    if not items:
        raise RuntimeError("No embedding records found. Run parse -> chunk -> embed first.")

    idx_sum, idx_full, E_sum, E_full, bm25 = build_indices(items)
    model_name, q_encoder = load_query_encoder()
    qv = encode_query(question, q_encoder)

    c_sum = dense_search(qv, idx_sum, k_dense_sum)
    c_full = dense_search(qv, idx_full, k_dense_full)
    c_sparse = sparse_search(question, bm25, k_sparse)

    fused = rrf([c_sum, c_full, c_sparse], weights=[0.9, 1.2, 0.8], rrf_k=60, base=60)
    fused = per_doc_cap(fused, items, per_doc)

    reranked = rerank(question, fused[:100], items, topn=20)
    final_indices = mmr_select(qv[0], reranked, E_full, lambda_mmr, min(final_k, len(reranked) or 0))

    sources_block, cmap = format_citations(final_indices, items)
    system, user = build_prompt(question, final_indices, items)
    out = call_gemini_via_litellm(system, user, timeout=timeout)

    # Cheap faithfulness: ensure most sentences have citations
    sentences = [s for s in re.split(r"(?<=[.!?])\s+", out) if s.strip()]
    cited = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
    if sentences and cited / len(sentences) < 0.6:
        out += "\n\nNote: Some statements may require verification; see Sources."

    return out, sources_block


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the Synapse index with hybrid RAG and generate with Gemini via LiteLLM")
    default_emb = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "embeddings.jsonl"))
    parser.add_argument("--question", type=str, required=False, help="User question")
    parser.add_argument("--embeddings", type=str, default=default_emb, help="Path to embeddings.jsonl")
    parser.add_argument("--timeout", type=int, default=60, help="LLM timeout (s)")
    parser.add_argument("--topk", type=int, default=8, help="Final K contexts")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    q = args.question or "What were the timing constraints for the version 2 Tensix L2 cache controller?"
    out, sources = answer(
        q,
        embeddings_path=os.path.abspath(args.embeddings),
        final_k=int(args.topk),
        timeout=int(args.timeout),
    )
    print(out.strip())
    print("\nSources:\n" + sources)
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())


