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
from typing import Dict, Iterable, List, Tuple, Optional

import numpy as np


# Optional rerankers
FLASHRANK_OK = False
try:  # pragma: no cover - optional dependency
    from flashrank import Ranker, RerankRequest  # type: ignore

    FLASHRANK_OK = True
except Exception:
    FLASHRANK_OK = False

CE = None
CROSS_ENCODER_CLASS = None

try:  # pragma: no cover - optional dependency
    from sentence_transformers import CrossEncoder  # type: ignore
    CROSS_ENCODER_CLASS = CrossEncoder
except Exception:
    CROSS_ENCODER_CLASS = None

def _get_cross_encoder():
    """Lazy loading of CrossEncoder to avoid HTTP 429 during module import."""
    global CE
    if CE is None and CROSS_ENCODER_CLASS is not None:
        try:
            CE = CROSS_ENCODER_CLASS("cross-encoder/ms-marco-MiniLM-L-6-v2")
        except Exception as e:
            print(f"⚠️  Could not load CrossEncoder model: {e}")
            CE = False  # Mark as failed to avoid retrying
    return CE if CE is not False else None


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


def rerank(query: str, candidate_indices: List[int], items: List[Dict[str, object]], topn: int = 20, chunk_index: Optional[Dict[str, Dict[str, object]]] = None) -> List[int]:
    if not candidate_indices:
        return []
    passages = []
    for i in candidate_indices:
        it = items[i]
        text = str(it["full_text"])[:2000]
        meta = dict(it.get("metadata", {}))
        is_csv = (meta.get("content_format") == "csv") or (str(it.get("source_type")) == "csv")
        preview = ""
        if is_csv and chunk_index is not None:
            ch = chunk_index.get(str(it.get("id")))
            if ch is not None:
                csv_text = str(ch.get("content", ""))
                preview = _csv_to_markdown_preview(csv_text)
        if not preview and str(it.get("source_path", "")).lower().endswith(".csv"):
            src = str(it.get("source_path"))
            if os.path.isfile(src):
                preview = _csv_file_to_markdown_preview(src)
        full_for_rank = (preview + "\n\n" + text) if preview else text
        passages.append({"id": str(i), "text": full_for_rank})
    if FLASHRANK_OK:
        try:
            ranker = Ranker()
            req = RerankRequest(query=query, passages=passages)
            ranked = ranker.rerank(req)
            return [int(getattr(p, "id", p.get("id") if isinstance(p, dict) else str(p))) for p in ranked][:topn]
        except Exception as e:
            print(f"⚠️  FlashRank failed, falling back to CrossEncoder: {e}")
            # Fall through to CrossEncoder fallback
    
    # Try CrossEncoder as fallback (lazy loaded)
    ce = _get_cross_encoder()
    if ce is not None:
        pairs = [(query, p["text"]) for p in passages]
        scores = ce.predict(pairs)
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
    
    # Group chunks by document
    doc_groups: Dict[str, List[Dict]] = {}
    for i in indices:
        it = items[i]
        src = str(it.get("source_path"))
        doc_name = src.split("/")[-1]  # Get just the filename
        meta = dict(it.get("metadata", {}))
        page = meta.get("page_number") or meta.get("slide_number")
        
        if doc_name not in doc_groups:
            doc_groups[doc_name] = []
        doc_groups[doc_name].append({"index": i, "page": page})
    
    # Create citations grouped by document
    lines: List[str] = []
    citation_num = 1
    
    for doc_name, chunks in doc_groups.items():
        # Collect all pages for this document
        pages = []
        chunk_indices = []
        for chunk in chunks:
            chunk_indices.append(chunk["index"])
            if chunk["page"]:
                pages.append(int(chunk["page"]))
        
        # Assign same citation number to all chunks from this document
        for chunk_idx in chunk_indices:
            mapping[chunk_idx] = f"[{citation_num}]"
        
        # Format the citation line
        if pages:
            pages_sorted = sorted(set(pages))  # Remove duplicates and sort
            if len(pages_sorted) == 1:
                page_info = f" (page {pages_sorted[0]})"
            elif len(pages_sorted) <= 5:
                page_info = f" (pages {', '.join(map(str, pages_sorted))})"
            else:
                # If too many pages, show range
                page_info = f" (pages {pages_sorted[0]}-{pages_sorted[-1]} and others)"
        else:
            page_info = ""
        
        lines.append(f"[{citation_num}] {doc_name}{page_info}")
        citation_num += 1
    
    return "\n".join(lines), mapping


def _csv_to_markdown_preview(csv_text: str, max_rows: int = 6, max_cols: int = 10, max_col_chars: int = 40) -> str:
    import csv
    from io import StringIO
    if not csv_text or not csv_text.strip():
        return ""
    f = StringIO(csv_text)
    try:
        reader = csv.reader(f)
        rows = list(reader)
    except Exception:
        return ""
    if not rows:
        return ""
    header = rows[0][:max_cols]
    data = rows[1:1 + max_rows]
    def clip(s: str) -> str:
        s = s.strip()
        return (s[:max_col_chars] + "…") if len(s) > max_col_chars else s
    header = [clip(h) for h in header]
    body = [[clip(c) for c in r[:max_cols]] for r in data]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in body:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _csv_file_to_markdown_preview(source_path: str, max_rows: int = 6, max_cols: int = 10, max_col_chars: int = 40) -> str:
    try:
        import pandas as pd  # type: ignore
    except Exception:
        return ""
    try:
        df = pd.read_csv(source_path, engine="python")
    except Exception:
        try:
            df = pd.read_csv(source_path, sep=None, engine="python")
        except Exception:
            try:
                df = pd.read_csv(source_path, delim_whitespace=True, engine="python")
            except Exception:
                return ""
    if df.empty:
        return ""
    df = df.iloc[:max_rows, :max_cols]
    def clip(s: str) -> str:
        s = str(s)
        return (s[:max_col_chars] + "…") if len(s) > max_col_chars else s
    header = [clip(c) for c in df.columns.tolist()]
    rows = [[clip(v) for v in row] for row in df.values.tolist()]
    lines = ["| " + " | ".join(header) + " |"]
    lines.append("| " + " | ".join(["---"] * len(header)) + " |")
    for r in rows:
        lines.append("| " + " | ".join(r) + " |")
    return "\n".join(lines)


def _pptx_slide_label(meta: Dict[str, object]) -> str:
    snum = meta.get("slide_number")
    return f"Slide {int(snum)}" if isinstance(snum, int) or (isinstance(snum, str) and str(snum).isdigit()) else "Slide"


def build_prompt(question: str, final_indices: List[int], items: List[Dict[str, object]],
                 chunk_index: Optional[Dict[str, Dict[str, object]]] = None) -> Tuple[str, str]:
    ctx_blocks: List[str] = []
    for n, i in enumerate(final_indices, start=1):
        it = items[i]
        text = str(it["full_text"]).strip()
        meta = dict(it.get("metadata", {}))
        is_csv = (meta.get("content_format") == "csv") or (str(it.get("source_type")) == "csv")
        is_pptx = str(it.get("source_type")) == "pptx"
        preview = ""
        if is_csv and chunk_index is not None:
            ch = chunk_index.get(str(it.get("id")))
            if ch is not None:
                csv_text = str(ch.get("content", ""))
                preview = _csv_to_markdown_preview(csv_text)
        if not preview and str(it.get("source_path", "")).lower().endswith(".csv"):
            src = str(it.get("source_path"))
            if os.path.isfile(src):
                preview = _csv_file_to_markdown_preview(src)
        if len(text) > 2200:
            text = text[:2200]
        if is_pptx:
            slide_title = _pptx_slide_label(meta)
            block = f"Source {n}: {slide_title}:\n{text}"
        elif preview:
            block = f"Source {n}: Table preview (first rows/cols):\n{preview}\n\nDetails: {text}"
        else:
            block = f"Source {n}: {text}"
        ctx_blocks.append(block)
    context = "\n\n".join(ctx_blocks)
    system = (
        "You are an engineering assistant. Keep your responses relevant and based on the context. "
        "Provide as much detail as possible but keep it concise. When asked for tables - reconstruct tables based on retrieved context. "
        "IMPORTANT: Never include any citation numbers, brackets like [1], [2], or references to 'Context Document X' in your response. "
        "Write naturally without any citation markers - the system will add citations separately. "
        "When interpreting OCR text from technical diagrams, be careful about common OCR errors: '2x' may appear as part of adjacent text, 'I2C' may appear as '12C', 'x4' formatting may be inconsistent. "
        "Always double-check technical specifications and component counts against the visual context when available. If the answer is not found, say you don't know."
    )
    user = f"Question: {question}\n\nContext:\n{context}\n\nInstructions:\n- Prefer drawing from the same source when multiple chunks are provided\n- Preserve list/table structure\n- Avoid speculation; do not use outside knowledge\n- Format your response with clear paragraphs and bullet points\n\nProvide a concise, well-formatted answer."
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


def _load_chunk_index(chunked_path: Optional[str]) -> Optional[Dict[str, Dict[str, object]]]:
    if not chunked_path:
        return None
    path = os.path.abspath(chunked_path)
    if not os.path.isfile(path):
        return None
    idx: Dict[str, Dict[str, object]] = {}
    for rec in read_jsonl(path):
        rid = str(rec.get("id"))
        if rid:
            idx[rid] = rec
    return idx


def _derive_chunked_path_from_embeddings(embeddings_path: str) -> Optional[str]:
    try:
        p = Path(embeddings_path)
        candidate = p.with_name("chunked.jsonl")
        return str(candidate)
    except Exception:
        return None


def extract_relevant_images(question: str, final_indices: List[int], items: List[Dict[str, object]]) -> List[str]:
    """Extract relevant image paths from retrieved context based on the question and context."""
    image_paths = []
    
    # Load the chunked file which has the correct image metadata
    try:
        chunked_items = []
        chunked_file = "artifacts/chunked_with_images.jsonl"
        if os.path.isfile(chunked_file):
            for rec in read_jsonl(chunked_file):
                chunked_items.append(rec)
    except Exception as e:
        print(f"Warning: Could not load chunked file for image extraction: {e}")
        chunked_items = []
    
    # First, collect images from the retrieved context directly
    for idx in final_indices:
        item = items[idx]
        source_type = str(item.get("source_type", ""))
        metadata = item.get("metadata", {})
        
        # Check if this is an extracted image chunk
        if source_type == "image" and metadata.get("content_format") == "extracted_image":
            image_path = str(metadata.get("image_path", ""))
            if image_path and os.path.isfile(image_path):
                if image_path not in image_paths:
                    image_paths.append(image_path)
        
        # Also check for original image files that were processed as images
        elif source_type == "image":
            source_path = str(item.get("source_path", ""))
            if source_path and os.path.isfile(source_path):
                if source_path not in image_paths:
                    image_paths.append(source_path)
    
    # Second, look for images in the same documents as retrieved text chunks
    # This helps find diagrams that are on the same pages as relevant text
    relevant_docs = set()
    relevant_pages = set()
    
    for idx in final_indices:
        item = items[idx]
        doc_id = str(item.get("document_id", ""))
        relevant_docs.add(doc_id)
        
        metadata = item.get("metadata", {})
        page_num = metadata.get("page_number")
        if page_num:
            source_path = str(item.get("source_path", ""))
            relevant_pages.add((source_path, page_num))
    
    # Find images from the same documents and pages using chunked data
    # Use chunked_items if available, otherwise fallback to items
    search_items = chunked_items if chunked_items else items
    
    for item in search_items:
        source_type = str(item.get("source_type", ""))
        if source_type == "image":
            item_doc_id = str(item.get("document_id", ""))
            metadata = item.get("metadata", {})
            
            # Check if this image is from a relevant document
            if item_doc_id in relevant_docs:
                # Check if it's from a relevant page or if it has matching keywords
                item_page = metadata.get("page_number")
                source_path = str(item.get("source_path", ""))
                
                page_key = (source_path, item_page)
                is_relevant_page = page_key in relevant_pages
                
                # Check for keyword matches with the question
                has_relevant_keywords = _has_relevant_technical_keywords(question, item)
                
                if is_relevant_page or has_relevant_keywords:
                    if metadata.get("content_format") == "extracted_image":
                        image_path = str(metadata.get("image_path", ""))
                    else:
                        image_path = source_path
                    
                    if image_path and os.path.isfile(image_path):
                        if image_path not in image_paths:
                            image_paths.append(image_path)
    
    # Third, heuristic matching based on specific technical terms
    # This helps find relevant diagrams even if they're not on the exact same page
    question_lower = question.lower()
    
    # Look for specific technical terms that indicate diagram relevance
    diagram_keywords = {
        "cluster": ["cluster", "cpu", "core", "architecture"],
        "ras": ["ras", "reliability", "error", "interrupt"],
        "cache": ["cache", "l1", "l2", "l3", "hierarchy"],
        "timer": ["timer", "wdt", "watchdog", "peripheral"],
        "architecture": ["architecture", "block", "diagram", "system"],
        "interface": ["interface", "connection", "protocol", "bus"],
        "ascalon": ["ascalon", "cpu", "risc-v", "vector"],
        "tensix": ["tensix", "neo", "auto", "blackhole", "grayskull", "wormhole"],
        "neo": ["neo", "tensix", "auto", "configuration", "automotive"]
    }
    
    for keyword_group, related_terms in diagram_keywords.items():
        if any(term in question_lower for term in related_terms):
            # Find images with matching technical keywords using chunked data
            for item in search_items:
                source_type = str(item.get("source_type", ""))
                if source_type == "image":
                    metadata = item.get("metadata", {})
                    technical_keywords = metadata.get("technical_keywords", [])
                    
                    # Also check content for technical terms
                    content = str(item.get("content", "")).lower()
                    
                    # Check if image has relevant technical keywords or content
                    has_keywords = any(keyword.lower() in [tk.lower() for tk in technical_keywords] 
                                     for keyword in related_terms)
                    has_content_match = any(keyword.lower() in content for keyword in related_terms)
                    
                    if has_keywords or has_content_match:
                        if metadata.get("content_format") == "extracted_image":
                            image_path = str(metadata.get("image_path", ""))
                        else:
                            image_path = str(item.get("source_path", ""))
                        
                        if image_path and os.path.isfile(image_path):
                            if image_path not in image_paths:
                                image_paths.append(image_path)
    
    # Limit to top 2 most relevant images to avoid overwhelming the response
    return image_paths[:2]  # Max 2 images per response


def _has_relevant_technical_keywords(question: str, image_item: Dict[str, object]) -> bool:
    """Check if an image has technical keywords relevant to the question."""
    question_lower = question.lower()
    metadata = image_item.get("metadata", {})
    
    # Check technical keywords
    technical_keywords = metadata.get("technical_keywords", [])
    for keyword in technical_keywords:
        if keyword.lower() in question_lower:
            return True
    
    # Check image content
    content = str(image_item.get("content", "")).lower()
    
    # Look for common technical terms that indicate relevance
    question_words = set(re.findall(r'\b\w+\b', question_lower))
    content_words = set(re.findall(r'\b\w+\b', content))
    
    # Technical terms that indicate diagram relevance
    important_terms = {
        "cpu", "core", "cluster", "cache", "timer", "interrupt", "ras", 
        "architecture", "block", "diagram", "interface", "connection",
        "ascalon", "risc-v", "vector", "pipeline", "peripheral", "bus"
    }
    
    # Check for significant overlap in important technical terms
    question_important = question_words & important_terms
    content_important = content_words & important_terms
    overlap = question_important & content_important
    
    return len(overlap) >= 2  # At least 2 overlapping important terms


def answer(
    question: str,
    embeddings_path: str,
    chunked_path: Optional[str] = None,
    k_dense_sum: int = 60,
    k_dense_full: int = 60,
    k_sparse: int = 60,
    per_doc: int = 8,
    final_k: int = 10,
    lambda_mmr: float = 0.8,
    timeout: int = 60,
    include_images: bool = True,
) -> Tuple[str, str, List[str]]:
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

    # Heuristic injection for CSV
    ql = question.lower()
    inject_indices: List[int] = []
    if any(tok in ql for tok in ["csv", "table", "column", "row", "requirements"]):
        toks = question.lower().split()
        scores = bm25.get_scores(toks)
        csv_indices = [i for i, it in enumerate(items) if str(it.get("source_type")) == "csv" or it.get("metadata", {}).get("content_format") == "csv"]
        top_csv = sorted(csv_indices, key=lambda i: scores[i], reverse=True)[:20]
        fused = list(dict.fromkeys(top_csv + fused))
    # PPTX injection
    ppt_inject: List[int] = []
    if any(tok in ql for tok in ["slide", "ppt", "pptx", "deck", "concept review"]):
        name_hints = ["alexandria concept review", "concept review", "alexandria"]
        for i, it in enumerate(items):
            sp = str(it.get("source_path", "")).lower()
            st = str(it.get("source_type", "")).lower()
            if st == "pptx" and any(h in sp for h in name_hints):
                ppt_inject.append(i)
        ppt_inject = ppt_inject[:30]
        if ppt_inject:
            fused = list(dict.fromkeys(ppt_inject + fused))

    fused = per_doc_cap(fused, items, per_doc)

    # Rerank
    chunk_index = None
    if chunked_path is None:
        derived = _derive_chunked_path_from_embeddings(embeddings_path)
        if derived and os.path.isfile(derived):
            chunk_index = _load_chunk_index(derived)
    else:
        chunk_index = _load_chunk_index(chunked_path)

    reranked = rerank(question, fused[:120], items, topn=40, chunk_index=chunk_index)

    # Coherence: prioritize the top document, then fill remainder
    if reranked:
        doc_counts: Dict[str, int] = {}
        for i in reranked:
            d = str(items[i]["document_id"])
            doc_counts[d] = doc_counts.get(d, 0) + 1
        top_doc = max(doc_counts.items(), key=lambda kv: kv[1])[0]
        top_doc_indices = [i for i in reranked if str(items[i]["document_id"]) == top_doc]
        others = [i for i in reranked if str(items[i]["document_id"]) != top_doc]
        final_indices = top_doc_indices[: min(len(top_doc_indices), final_k)]
        for i in others:
            if len(final_indices) >= final_k:
                break
            final_indices.append(i)
    else:
        final_indices = []

    sources_block, cmap = format_citations(final_indices, items)
    system, user = build_prompt(question, final_indices, items, chunk_index)
    out = call_gemini_via_litellm(system, user, timeout=timeout)
    
    # Extract relevant images from the retrieved context
    relevant_images = []
    if include_images:
        relevant_images = extract_relevant_images(question, final_indices, items)

    # Citation verification note removed since we handle citations separately through sources_block
    # sentences = [s for s in re.split(r"(?<=[.!?])\s+", out) if s.strip()]
    # cited = sum(1 for s in sentences if re.search(r"\[\d+\]", s))
    # if sentences and cited / len(sentences) < 0.6:
    #     out += "\n\nNote: Some statements may require verification; see Sources."

    return out, sources_block, relevant_images


def list_available_knowledge_bases(artifacts_dir: str) -> List[Dict[str, str]]:
    """List all available knowledge bases (folder-based embeddings)."""
    import glob
    
    # Find all embedding files
    pattern = os.path.join(artifacts_dir, "embedded_*.jsonl")
    embedding_files = glob.glob(pattern)
    
    knowledge_bases = []
    for embed_file in sorted(embedding_files):
        basename = os.path.basename(embed_file)
        if basename.startswith("embedded_") and basename.endswith(".jsonl"):
            folder_name = basename[9:-6]  # Remove "embedded_" and ".jsonl"
            
            # Find corresponding chunked file
            chunked_file = os.path.join(artifacts_dir, f"chunked_{folder_name}.jsonl")
            
            # Get file stats for additional info
            try:
                file_stats = os.stat(embed_file)
                file_size = file_stats.st_size
                # Count chunks by reading the file
                chunk_count = 0
                try:
                    with open(embed_file, 'r') as f:
                        for _ in f:
                            chunk_count += 1
                except:
                    chunk_count = 0
            except:
                file_size = 0
                chunk_count = 0
            
            knowledge_bases.append({
                'name': folder_name,
                'display_name': folder_name.replace('_', ' ').replace('hash_', '#'),
                'embeddings_path': embed_file,
                'chunked_path': chunked_file if os.path.exists(chunked_file) else None,
                'file_size': file_size,
                'chunk_count': chunk_count
            })
    
    return knowledge_bases


def interactive_knowledge_base_selector(artifacts_dir: str) -> Optional[Dict[str, str]]:
    """Interactive knowledge base selection with enhanced display."""
    knowledge_bases = list_available_knowledge_bases(artifacts_dir)
    
    if not knowledge_bases:
        print("❌ No knowledge bases found. Run folder-based initialization first:")
        print("   python initialize_folders.py")
        return None
    
    print("\n" + "="*80)
    print("📚 AVAILABLE KNOWLEDGE BASES")
    print("="*80)
    
    for i, kb in enumerate(knowledge_bases, 1):
        size_mb = kb['file_size'] / (1024 * 1024) if kb['file_size'] > 0 else 0
        chunk_info = f"{kb['chunk_count']:,} chunks" if kb['chunk_count'] > 0 else "unknown chunks"
        size_info = f"{size_mb:.1f} MB" if size_mb > 0 else "unknown size"
        
        print(f"  {i:2d}. 📁 {kb['display_name']}")
        print(f"      ID: {kb['name']}")
        print(f"      📊 {chunk_info}, {size_info}")
        print(f"      📄 {os.path.basename(kb['embeddings_path'])}")
        if kb['chunked_path']:
            print(f"      🧩 {os.path.basename(kb['chunked_path'])}")
        print()
    
    print(f"  0.  🔙 Use default knowledge base")
    print("="*80)
    
    while True:
        try:
            choice = input(f"Select knowledge base (0-{len(knowledge_bases)}): ").strip()
            
            if choice == "0":
                return None  # Use default
            
            choice_num = int(choice)
            if 1 <= choice_num <= len(knowledge_bases):
                selected_kb = knowledge_bases[choice_num - 1]
                print(f"✅ Selected: {selected_kb['display_name']}")
                return selected_kb
            else:
                print(f"❌ Please enter a number between 0 and {len(knowledge_bases)}")
        except ValueError:
            print("❌ Please enter a valid number")
        except (EOFError, KeyboardInterrupt):
            print("\n❌ Selection cancelled")
            return None


def main(argv: List[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Query the Synapse index with hybrid RAG and generate with Gemini via LiteLLM")
    default_emb = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "embedded_with_images.npz"))
    parser.add_argument("--question", type=str, required=False, help="User question")
    parser.add_argument("--embeddings", type=str, default=default_emb, help="Path to embeddings.jsonl or knowledge base name")
    # --chunked is now optional; if omitted, derived from embeddings path
    parser.add_argument("--chunked", type=str, default=None, help="Path to chunked.jsonl (optional; auto-derived)")
    parser.add_argument("--list-kb", action="store_true", help="List available knowledge bases and exit")
    parser.add_argument("--kb", type=str, help="Use specific knowledge base by name (e.g., 'Confluence_IPS', 'Aahan_Notes')")
    parser.add_argument("--select-kb", action="store_true", help="Interactively select knowledge base from available options")
    parser.add_argument("--timeout", type=int, default=60, help="LLM timeout (s)")
    parser.add_argument("--topk", type=int, default=10, help="Final K contexts")
    parser.add_argument("--verbose", action="store_true")
    args = parser.parse_args(argv)

    # Handle knowledge base listing
    if args.list_kb:
        artifacts_dir = os.path.dirname(os.path.abspath(args.embeddings))
        knowledge_bases = list_available_knowledge_bases(artifacts_dir)
        
        if not knowledge_bases:
            print("No knowledge bases found. Run folder-based parsing and embedding first:")
            print("  python initialize_folders.py")
            return 1
        
        print("Available Knowledge Bases:")
        print("=" * 50)
        for kb in knowledge_bases:
            size_mb = kb['file_size'] / (1024 * 1024) if kb['file_size'] > 0 else 0
            chunk_info = f"{kb['chunk_count']:,} chunks" if kb['chunk_count'] > 0 else "unknown chunks"
            size_info = f"{size_mb:.1f} MB" if size_mb > 0 else "unknown size"
            
            print(f"📚 {kb['display_name']} ({kb['name']})")
            print(f"   📊 {chunk_info}, {size_info}")
            print(f"   📄 Embeddings: {os.path.basename(kb['embeddings_path'])}")
            if kb['chunked_path']:
                print(f"   🧩 Chunks: {os.path.basename(kb['chunked_path'])}")
            print()
        
        print("Usage:")
        print(f"  python pipeline/query.py --kb \"{knowledge_bases[0]['name']}\" --question \"Your question\"")
        print(f"  python pipeline/query.py --select-kb --question \"Your question\"")
        return 0
    
    # Handle interactive knowledge base selection
    if args.select_kb:
        artifacts_dir = os.path.dirname(os.path.abspath(args.embeddings))
        selected_kb = interactive_knowledge_base_selector(artifacts_dir)
        
        if selected_kb is None:
            print("Using default knowledge base...")
        else:
            # Override the kb argument with the selected knowledge base
            args.kb = selected_kb['name']
    
    # Handle knowledge base selection
    embeddings_path = args.embeddings
    chunked_path = args.chunked
    
    if args.kb:
        # Find the specified knowledge base
        artifacts_dir = os.path.dirname(os.path.abspath(args.embeddings))
        knowledge_bases = list_available_knowledge_bases(artifacts_dir)
        
        selected_kb = None
        for kb in knowledge_bases:
            if kb['name'] == args.kb or kb['display_name'].lower() == args.kb.lower():
                selected_kb = kb
                break
        
        if not selected_kb:
            print(f"Knowledge base '{args.kb}' not found. Available options:")
            for kb in knowledge_bases:
                print(f"  - {kb['name']} ({kb['display_name']})")
            return 1
        
        embeddings_path = selected_kb['embeddings_path']
        chunked_path = selected_kb['chunked_path']
        print(f"🔍 Using knowledge base: {selected_kb['display_name']}")
        print(f"📄 Embeddings: {os.path.basename(embeddings_path)}")
        if chunked_path:
            print(f"🧩 Chunks: {os.path.basename(chunked_path)}")
        print()

    q = args.question or "What were the timing constraints for the version 2 Tensix L2 cache controller?"
    out, sources, images = answer(
        q,
        embeddings_path=os.path.abspath(embeddings_path),
        chunked_path=os.path.abspath(chunked_path) if chunked_path else None,
        final_k=int(args.topk),
        timeout=int(args.timeout),
    )
    print(out.strip())
    print("\nSources:\n" + sources)
    if images:
        print(f"\nRelevant Images ({len(images)}):")
        for i, img_path in enumerate(images, 1):
            print(f"  [{i}] {img_path}")
    return 0


if __name__ == "__main__":
    import sys

    raise SystemExit(main())


