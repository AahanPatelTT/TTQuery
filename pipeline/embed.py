#!/usr/bin/env python3
"""
Embedding step for the Synapse pipeline (Parse -> Chunk -> Embed -> Retrieve -> Generate)

This script reads chunked JSONL and produces multi-vector representations per chunk:
- Summary Vector: embedding of a short, concise summary of the chunk (broad retrieval)
- Full-Content Vector: embedding of the full, linearized text (fine-grained retrieval)

Key features
- Table linearization: CSV tables are converted into sentence-based summaries with limited
  row sampling and column descriptions before embedding.
- Image augmentation: image captions are augmented with nearby textual context from the
  same page to improve retrieval quality for figures/diagrams.
- Providers: local SentenceTransformers by default; optional OpenAI embeddings via API.

Output
- JSONL file with, per chunk id: summary_text, full_text, embedding_summary, embedding_full,
  along with original metadata for traceability.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import pickle
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Tuple


# -------------------------------
# Cache management
# -------------------------------


@dataclass
class EmbedCacheEntry:
    """Cache entry storing input file metadata and embedding results."""
    input_file: str
    input_mtime: float
    input_size: int
    config_hash: str  # Hash of embedding configuration
    embeddings: List[Dict[str, object]]


def get_embed_cache_path(output_path: str) -> str:
    """Get cache file path based on output path."""
    output_dir = os.path.dirname(output_path)
    cache_name = os.path.splitext(os.path.basename(output_path))[0] + "_embed_cache.pkl"
    return os.path.join(output_dir, cache_name)


def compute_embed_config_hash(provider: str, model_name: str, summary_mode: str) -> str:
    """Compute hash of embedding configuration for cache validation."""
    import hashlib
    config_str = f"{provider}:{model_name}:{summary_mode}"
    return hashlib.sha256(config_str.encode()).hexdigest()[:16]


def load_embed_cache(cache_path: str) -> EmbedCacheEntry:
    """Load cache from file. Returns empty cache if file doesn't exist or is invalid."""
    if not os.path.isfile(cache_path):
        return EmbedCacheEntry("", 0.0, 0, "", [])
    
    try:
        with open(cache_path, "rb") as f:
            return pickle.load(f)
    except Exception as e:
        logging.warning("Failed to load embed cache from %s: %s", cache_path, e)
        return EmbedCacheEntry("", 0.0, 0, "", [])


def save_embed_cache(cache_entry: EmbedCacheEntry, cache_path: str) -> None:
    """Save cache entry to file."""
    try:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        with open(cache_path, "wb") as f:
            pickle.dump(cache_entry, f)
        logging.debug("Saved embed cache to %s", cache_path)
    except Exception as e:
        logging.warning("Failed to save embed cache to %s: %s", cache_path, e)


def is_embed_cache_valid(input_path: str, cache_entry: EmbedCacheEntry, config_hash: str) -> bool:
    """Check if cache is valid for the given input file and configuration."""
    if not cache_entry.input_file or cache_entry.input_file != input_path:
        return False
    
    if cache_entry.config_hash != config_hash:
        return False
    
    try:
        stat = os.stat(input_path)
        return (stat.st_mtime == cache_entry.input_mtime and 
                stat.st_size == cache_entry.input_size)
    except OSError:
        return False


def read_jsonl(path: str) -> Iterator[Dict[str, object]]:
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def write_jsonl(records: Iterable[Dict[str, object]], path: str) -> int:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    count = 0
    with open(path, "w", encoding="utf-8") as f:
        for rec in records:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            count += 1
    return count


# -------------------------------
# Linearization helpers
# -------------------------------


def summarize_csv_table(csv_text: str, max_sample_rows: int = 10) -> str:
    """Linearize a CSV into compact, sentence-based text.

    Strategy:
    - Identify columns and row count
    - Include a terse description of columns
    - Sample up to N rows and render each as a compact key=value list
    """
    lines = [ln for ln in csv_text.splitlines() if ln.strip()]
    if not lines:
        return "Empty table."
    header = [h.strip() for h in lines[0].split(",")]
    rows = [r.split(",") for r in lines[1:]]

    col_desc = ", ".join(header[:20])  # cap to avoid overlong headers
    parts: List[str] = []
    parts.append(f"Table with {len(rows)} rows and {len(header)} columns: {col_desc}.")
    if rows:
        sample = rows[:max_sample_rows]
        for idx, r in enumerate(sample, start=1):
            kv = ", ".join(
                f"{header[i]}={r[i].strip()}" for i in range(min(len(header), len(r))) if header[i]
            )
            parts.append(f"Row {idx}: {kv}.")
        if len(rows) > max_sample_rows:
            parts.append(f"… {len(rows) - max_sample_rows} more rows not shown.")
    return " " .join(parts)


def build_page_index(chunks: List[Dict[str, object]]) -> Dict[Tuple[str, Optional[int]], List[Dict[str, object]]]:
    """Group chunks by (source_path, page_number) for local page context lookups."""
    index: Dict[Tuple[str, Optional[int]], List[Dict[str, object]]] = {}
    for ch in chunks:
        src = str(ch.get("source_path", ""))
        meta = dict(ch.get("metadata", {}) or {})
        page = meta.get("page_number")
        key = (src, int(page) if page is not None else None)
        index.setdefault(key, []).append(ch)
    return index


def surrounding_text_for_image(
    chunk: Dict[str, object], page_index: Dict[Tuple[str, Optional[int]], List[Dict[str, object]]], max_chars: int = 600
) -> str:
    """Extract nearby text from the same page as contextual augmentation for images."""
    src = str(chunk.get("source_path", ""))
    meta = dict(chunk.get("metadata", {}) or {})
    page = meta.get("page_number")
    key = (src, int(page) if page is not None else None)
    neighbors = page_index.get(key, [])
    # Prefer narrative and headings from neighbors
    texts: List[str] = []
    for n in neighbors:
        if n.get("id") == chunk.get("id"):
            continue
        nmeta = dict(n.get("metadata", {}) or {})
        etype = str(nmeta.get("element_type") or "")
        if etype in {"Title", "NarrativeText", "List", "Paragraph"} or n.get("source_type") in {"md", "txt"}:
            content = str(n.get("content", "") or "").strip()
            if content:
                texts.append(content)
        if sum(len(t) for t in texts) > max_chars:
            break
    merged = "\n\n".join(texts)
    if len(merged) > max_chars:
        merged = merged[:max_chars] + "…"
    return merged


def build_full_text(chunk: Dict[str, object], page_index: Dict[Tuple[str, Optional[int]], List[Dict[str, object]]]) -> str:
    """Return the linearized full-text representation of a chunk for embedding."""
    content = str(chunk.get("content", "") or "")
    meta = dict(chunk.get("metadata", {}) or {})
    source_type = str(chunk.get("source_type", ""))

    # Tables
    if meta.get("content_format") == "csv" or source_type == "csv":
        return summarize_csv_table(content)

    # Images: combine caption + OCR + neighboring text
    if source_type == "image":
        caption = ""
        ocr = ""
        if content:
            # Split our combined content form into caption and OCR if present
            if content.startswith("Caption:"):
                parts = content.split("OCR:", 1)
                caption = parts[0].replace("Caption:", "").strip()
                ocr = parts[1].strip() if len(parts) > 1 else ""
            else:
                ocr = content
        surround = surrounding_text_for_image(chunk, page_index)
        pieces = []
        if caption:
            pieces.append(f"Figure caption: {caption}.")
        if ocr:
            pieces.append(f"Figure text (OCR): {ocr}.")
        if surround:
            pieces.append(f"Nearby context: {surround}")
        return " \n".join(pieces) if pieces else content

    # Markdown headings: include path in text for better retrieval
    heading_path = meta.get("heading_path") or []
    if heading_path and isinstance(heading_path, list):
        prefix = " > ".join([str(h) for h in heading_path if h])
        return f"Section: {prefix}.\n{content}" if content else f"Section: {prefix}."

    return content


def heuristic_summary(chunk: Dict[str, object], full_text: str, max_chars: int = 280) -> str:
    meta = dict(chunk.get("metadata", {}) or {})
    source_type = str(chunk.get("source_type", ""))
    if meta.get("content_format") == "csv" or source_type == "csv":
        cols = meta.get("columns") or []
        nrows = meta.get("num_rows")
        if isinstance(cols, list) and cols:
            col_list = ", ".join([str(c) for c in cols[:10]])
        else:
            col_list = "table columns"
        nrows_str = f"{int(nrows)}" if isinstance(nrows, int) else "multiple"
        return f"This table summarizes {nrows_str} rows across columns {col_list}."
    if source_type == "image":
        # Prefer the caption if present
        if "Caption:" in full_text:
            first = full_text.split("\n", 1)[0]
            return first.replace("Figure caption:", "This figure").strip().rstrip(".") + "."
        return "This figure provides visual information relevant to the surrounding section."
    # Default: first sentence/line trimmed
    first_line = full_text.strip().split("\n")[0]
    if len(first_line) > max_chars:
        first_line = first_line[:max_chars].rstrip() + "…"
    return first_line


# -------------------------------
# Embedding providers
# -------------------------------


class LocalEmbedder:
    def __init__(self, model_name: str = "BAAI/bge-large-en-v1.5") -> None:
        try:
            from sentence_transformers import SentenceTransformer  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "sentence-transformers is required for local embeddings. 'pip install sentence-transformers'"
            ) from exc

        self.model = SentenceTransformer(model_name)

    def embed(self, texts: List[str]) -> List[List[float]]:
        vectors = self.model.encode(texts, normalize_embeddings=True, show_progress_bar=False)
        return [list(map(float, v)) for v in vectors]


class OpenAIEmbedder:
    def __init__(self, model_name: str = "text-embedding-3-small") -> None:
        try:
            from openai import OpenAI  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("openai>=1.x package is required for OpenAI embeddings.") from exc
        self.client = OpenAI()
        self.model_name = model_name

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        resp = self.client.embeddings.create(model=self.model_name, input=texts)
        return [list(map(float, d.embedding)) for d in resp.data]


class ColBERTEmbedder:
    def __init__(self, model_name: str = "colbert-ir/colbertv2.0", max_length: int = 256) -> None:
        # Prefer true ColBERT if available; otherwise fall back to per-token HF embeddings
        self.max_length = int(max_length)
        self._use_true_colbert = False
        try:
            # Optional dependency: colbert-ai
            from colbert.infra import ColBERTConfig  # type: ignore
            from colbert.modeling.checkpoint import Checkpoint  # type: ignore
            import torch  # type: ignore

            self._torch = torch
            # Load ColBERT checkpoint and tokenizer
            self._checkpoint = Checkpoint(model_name)
            self._colbert = self._checkpoint.get_model()
            self._tokenizer = self._checkpoint.tokenizer
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._colbert.to(self._device)
            self._colbert.eval()
            self._use_true_colbert = True
        except Exception:
            # Fallback: HuggingFace AutoModel token embeddings (not projected to 128-d)
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            import torch  # type: ignore

            self._torch = torch
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            self._model = AutoModel.from_pretrained(model_name)
            self._device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model.to(self._device)
            self._model.eval()

    def embed_token_vectors(self, texts: List[str]) -> List[List[List[float]]]:
        """Return per-token embeddings for each input text.

        Output shape: [batch] -> [num_tokens, dim]
        - Special tokens are removed (CLS/SEP) where identifiable.
        - Vectors are L2-normalized per token.
        """
        if not texts:
            return []

        torch = self._torch
        results: List[List[List[float]]] = []

        if getattr(self, "_use_true_colbert", False):
            # True ColBERT path
            tokenizer = self._tokenizer
            model = self._colbert
            device = self._device
            with torch.no_grad():
                enc = tokenizer(
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                    return_tensors="pt",
                )
                enc = {k: v.to(device) for k, v in enc.items()}
                # ColBERT exposes doc encoding via model.doc; yields [B, L, D]
                # Some versions expose model.forward; use getattr defensively
                doc_encode_fn = getattr(model, "doc", None)
                if doc_encode_fn is None:
                    # Fallback to calling model with is_query=False if available
                    outputs = model(**enc)  # type: ignore
                else:
                    outputs = doc_encode_fn(**enc)  # type: ignore

                # Normalize per token
                if isinstance(outputs, tuple):
                    token_emb = outputs[0]
                else:
                    token_emb = outputs
                # token_emb: [B, L, D]
                token_emb = token_emb
                attn = enc.get("attention_mask")  # [B, L]
                norms = torch.norm(token_emb, dim=-1, keepdim=True).clamp(min=1e-12)
                token_emb = token_emb / norms

                for i in range(token_emb.shape[0]):
                    mask = attn[i] if attn is not None else None
                    vecs = token_emb[i]
                    # Remove [CLS]/[SEP] if token ids are available
                    input_ids = enc.get("input_ids")
                    keep: List[int] = []
                    for j in range(vecs.shape[0]):
                        if mask is not None and mask[j].item() == 0:
                            continue
                        if input_ids is not None:
                            tid = int(input_ids[i, j].item())
                            if tid in {tokenizer.cls_token_id, tokenizer.sep_token_id}:
                                continue
                        keep.append(j)
                    picked = vecs[keep] if keep else vecs[:0]
                    results.append([[float(x) for x in row.tolist()] for row in picked])
            return results

        # HuggingFace fallback path
        tokenizer = self._tokenizer
        model = self._model
        device = self._device

        with torch.no_grad():
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            token_emb = out.last_hidden_state  # [B, L, H]
            attn = enc.get("attention_mask")
            norms = torch.norm(token_emb, dim=-1, keepdim=True).clamp(min=1e-12)
            token_emb = token_emb / norms

            for i in range(token_emb.shape[0]):
                mask = attn[i] if attn is not None else None
                vecs = token_emb[i]
                input_ids = enc.get("input_ids")
                keep: List[int] = []
                for j in range(vecs.shape[0]):
                    if mask is not None and mask[j].item() == 0:
                        continue
                    if input_ids is not None:
                        tid = int(input_ids[i, j].item())
                        if tid in {tokenizer.cls_token_id, tokenizer.sep_token_id}:
                            continue
                    keep.append(j)
                picked = vecs[keep] if keep else vecs[:0]
                results.append([[float(x) for x in row.tolist()] for row in picked])
        return results

    def embed_pooled(self, texts: List[str]) -> List[List[float]]:
        """Return a single dense vector per text by mean-pooling token vectors.
        This preserves compatibility with consumers expecting single vectors.
        """
        import numpy as np  # type: ignore

        token_vectors = self.embed_token_vectors(texts)
        pooled: List[List[float]] = []
        for tv in token_vectors:
            if not tv:
                pooled.append([])
                continue
            arr = np.asarray(tv, dtype=np.float32)  # [L, D]
            mean_vec = arr.mean(axis=0)
            # L2 normalize
            denom = float(np.linalg.norm(mean_vec) + 1e-12)
            mean_vec = mean_vec / denom
            pooled.append([float(x) for x in mean_vec.tolist()])
        return pooled


class BertEmbedder:
    def __init__(self, model_name: str = "bert-base-uncased", max_length: int = 256) -> None:
        try:
            from transformers import AutoModel, AutoTokenizer  # type: ignore
            import torch  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "transformers and torch are required for BERT embeddings. 'pip install transformers torch'"
            ) from exc

        self._torch = torch
        self._tokenizer = AutoTokenizer.from_pretrained(model_name)
        self._model = AutoModel.from_pretrained(model_name)
        self._device = "cuda" if torch.cuda.is_available() else "cpu"
        self._model.to(self._device)
        self._model.eval()
        self.max_length = int(max_length)

    def embed(self, texts: List[str]) -> List[List[float]]:
        if not texts:
            return []
        torch = self._torch
        tokenizer = self._tokenizer
        model = self._model
        device = self._device
        with torch.no_grad():
            enc = tokenizer(
                texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            out = model(**enc)
            token_emb = out.last_hidden_state  # [B, L, H]
            attn = enc.get("attention_mask")  # [B, L]
            # Mean pool over tokens with attention mask
            if attn is not None:
                mask = attn.unsqueeze(-1)  # [B, L, 1]
                summed = (token_emb * mask).sum(dim=1)  # [B, H]
                denom = mask.sum(dim=1).clamp(min=1e-6)  # [B, 1]
                pooled = summed / denom
            else:
                pooled = token_emb.mean(dim=1)
            # L2 normalize
            norms = pooled.norm(dim=-1, keepdim=True).clamp(min=1e-12)
            pooled = pooled / norms
            return [list(map(float, v.tolist())) for v in pooled]


def _infer_doc_prefix(model_name: str) -> str:
    name = model_name.lower()
    # Common instruction-tuned retrieval models expect 'passage:' for docs
    if "e5" in name or "bge" in name:
        return "passage: "
    return ""


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Compute multi-vector embeddings (summary + full) for chunks")
    default_input = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "chunked.jsonl"))
    default_output = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "embeddings.jsonl"))
    parser.add_argument("--input", type=str, default=default_input, help="Input chunked JSONL path")
    parser.add_argument("--output", type=str, default=default_output, help="Output embeddings JSONL path")
    parser.add_argument("--provider", type=str, choices=["local", "openai", "colbert", "bert"], default="local", help="Embedding provider")
    parser.add_argument("--embed-model", type=str, default="BAAI/bge-large-en-v1.5", help="Local model or OpenAI model name")
    parser.add_argument("--summary-mode", type=str, choices=["heuristic", "llm"], default="heuristic", help="How to produce summary text")
    parser.add_argument("--openai-embed-model", type=str, default="text-embedding-3-small", help="OpenAI embedding model if provider=openai")
    parser.add_argument("--colbert-model", type=str, default="colbert-ir/colbertv2.0", help="HuggingFace or ColBERT checkpoint name for provider=colbert")
    parser.add_argument("--colbert-max-length", type=int, default=256, help="Max sequence length for ColBERT token embeddings")
    parser.add_argument("--bert-model", type=str, default="bert-base-uncased", help="HuggingFace model name for provider=bert")
    parser.add_argument("--bert-max-length", type=int, default=256, help="Max sequence length for BERT fallback embeddings")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    parser.add_argument(
        "--no-cache",
        action="store_true",
        help="Disable caching - recompute embeddings even if input hasn't changed",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        help="Custom cache file path (default: auto-generated based on output path)",
    )
    args = parser.parse_args(argv)

    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)
    if not os.path.isfile(input_path):
        logging.error("Input file does not exist: %s", input_path)
        return 2

    # Initialize cache
    cache_path = args.cache_path or get_embed_cache_path(output_path)
    
    # Determine model name for config hash
    if args.provider == "local":
        model_name = args.embed_model
    elif args.provider == "openai":
        model_name = args.openai_embed_model
    elif args.provider == "colbert":
        model_name = args.colbert_model
    else:  # bert
        model_name = args.bert_model
    
    config_hash = compute_embed_config_hash(args.provider, model_name, args.summary_mode)
    
    # Check cache
    if not args.no_cache:
        cache_entry = load_embed_cache(cache_path)
        if is_embed_cache_valid(input_path, cache_entry, config_hash):
            logging.info("✅ Cache hit! Using cached embeddings from %s", cache_path)
            embeddings = cache_entry.embeddings
            written = write_jsonl(embeddings, output_path)
            logging.info("Wrote %d cached embeddings to %s", written, output_path)
            
            # Print summary with big font
            print("\n" + "="*80)
            print("███████╗███╗   ███╗██████╗ ███████╗██████╗ ██████╗ ██╗███╗   ██╗ ██████╗ ")
            print("██╔════╝████╗ ████║██╔══██╗██╔════╝██╔══██╗██╔══██╗██║████╗  ██║██╔════╝ ")
            print("█████╗  ██╔████╔██║██████╔╝█████╗  ██║  ██║██║  ██║██║██╔██╗ ██║██║  ███╗")
            print("██╔══╝  ██║╚██╔╝██║██╔══██╗██╔══╝  ██║  ██║██║  ██║██║██║╚██╗██║██║   ██║")
            print("███████╗██║ ╚═╝ ██║██████╔╝███████╗██████╔╝██████╔╝██║██║ ╚████║╚██████╔╝")
            print("╚══════╝╚═╝     ╚═╝╚═════╝ ╚══════╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ ")
            print("="*80)
            print("🔮 EMBEDDING COMPLETED (FROM CACHE)")
            print("="*80)
            print(f"📁 INPUT FILE: {input_path}")
            print(f"📄 OUTPUT FILE: {output_path}")
            print(f"🔢 TOTAL EMBEDDINGS: {len(embeddings):,}")
            print(f"⚡ STATUS: Cache hit - no processing needed")
            print("✅ EMBEDDING COMPLETED SUCCESSFULLY!")
            print("="*80 + "\n")
            return 0

    chunks = list(read_jsonl(input_path))
    if not chunks:
        logging.warning("No chunks found in input.")

    # Build page index for image context augmentation
    page_index = build_page_index(chunks)

    # Prepare embedder
    if args.provider == "local":
        embedder = LocalEmbedder(model_name=args.embed_model)
        doc_prefix = _infer_doc_prefix(args.embed_model)
    elif args.provider == "openai":
        embedder = OpenAIEmbedder(model_name=args.openai_embed_model)
        doc_prefix = ""
    elif args.provider == "colbert":
        embedder = ColBERTEmbedder(model_name=args.colbert_model, max_length=int(args.colbert_max_length))
        doc_prefix = ""
    else:  # bert
        embedder = BertEmbedder(model_name=args.bert_model, max_length=int(args.bert_max_length))
        doc_prefix = ""

    outputs: List[Dict[str, object]] = []
    batch_summary: List[str] = []
    batch_full: List[str] = []
    batch_meta: List[Dict[str, object]] = []

    for ch in chunks:
        full_text = build_full_text(ch, page_index)
        # Produce summary text
        if args.summary_mode == "heuristic":
            summary_text = heuristic_summary(ch, full_text)
        else:
            # For now fallback to heuristic; can be swapped to an LLM call if desired
            summary_text = heuristic_summary(ch, full_text)

        batch_summary.append((doc_prefix + summary_text) if doc_prefix else summary_text)
        batch_full.append((doc_prefix + full_text) if doc_prefix else full_text)
        batch_meta.append(
            {
                "id": ch.get("id"),
                "document_id": ch.get("document_id"),
                "source_path": ch.get("source_path"),
                "source_type": ch.get("source_type"),
                "metadata": ch.get("metadata", {}),
                "summary_text": summary_text,
                "full_text": full_text,
            }
        )

        # Flush in moderate batches for memory safety
        if len(batch_meta) >= 64:
            if args.provider == "colbert":
                # Native multi-vector per token + pooled vectors for compatibility
                emb_sum_mv = embedder.embed_token_vectors(batch_summary)  # type: ignore[attr-defined]
                emb_full_mv = embedder.embed_token_vectors(batch_full)  # type: ignore[attr-defined]
                emb_sum = embedder.embed_pooled(batch_summary)  # type: ignore[attr-defined]
                emb_full = embedder.embed_pooled(batch_full)  # type: ignore[attr-defined]
                for meta, v_s, v_f, mv_s, mv_f in zip(batch_meta, emb_sum, emb_full, emb_sum_mv, emb_full_mv):
                    outputs.append({**meta, "embedding_summary": v_s, "embedding_full": v_f, "embedding_summary_mv": mv_s, "embedding_full_mv": mv_f})
            else:
                emb_sum = embedder.embed(batch_summary)
                emb_full = embedder.embed(batch_full)
                for meta, v_s, v_f in zip(batch_meta, emb_sum, emb_full):
                    outputs.append({**meta, "embedding_summary": v_s, "embedding_full": v_f})
            batch_summary.clear()
            batch_full.clear()
            batch_meta.clear()

    # Flush remaining
    if batch_meta:
        if args.provider == "colbert":
            emb_sum_mv = embedder.embed_token_vectors(batch_summary)  # type: ignore[attr-defined]
            emb_full_mv = embedder.embed_token_vectors(batch_full)  # type: ignore[attr-defined]
            emb_sum = embedder.embed_pooled(batch_summary)  # type: ignore[attr-defined]
            emb_full = embedder.embed_pooled(batch_full)  # type: ignore[attr-defined]
            for meta, v_s, v_f, mv_s, mv_f in zip(batch_meta, emb_sum, emb_full, emb_sum_mv, emb_full_mv):
                outputs.append({**meta, "embedding_summary": v_s, "embedding_full": v_f, "embedding_summary_mv": mv_s, "embedding_full_mv": mv_f})
        else:
            emb_sum = embedder.embed(batch_summary)
            emb_full = embedder.embed(batch_full)
            for meta, v_s, v_f in zip(batch_meta, emb_sum, emb_full):
                outputs.append({**meta, "embedding_summary": v_s, "embedding_full": v_f})

    written = write_jsonl(outputs, output_path)
    
    # Save to cache
    if not args.no_cache:
        try:
            stat = os.stat(input_path)
            new_cache_entry = EmbedCacheEntry(
                input_file=input_path,
                input_mtime=stat.st_mtime,
                input_size=stat.st_size,
                config_hash=config_hash,
                embeddings=outputs
            )
            save_embed_cache(new_cache_entry, cache_path)
        except Exception as e:
            logging.warning("Failed to save cache: %s", e)
    
    # Print summary with big font
    print("\n" + "="*80)
    print("███████╗███╗   ███╗██████╗ ███████╗██████╗ ██████╗ ██╗███╗   ██╗ ██████╗ ")
    print("██╔════╝████╗ ████║██╔══██╗██╔════╝██╔══██╗██╔══██╗██║████╗  ██║██╔════╝ ")
    print("█████╗  ██╔████╔██║██████╔╝█████╗  ██║  ██║██║  ██║██║██╔██╗ ██║██║  ███╗")
    print("██╔══╝  ██║╚██╔╝██║██╔══██╗██╔══╝  ██║  ██║██║  ██║██║██║╚██╗██║██║   ██║")
    print("███████╗██║ ╚═╝ ██║██████╔╝███████╗██████╔╝██████╔╝██║██║ ╚████║╚██████╔╝")
    print("╚══════╝╚═╝     ╚═╝╚═════╝ ╚══════╝╚═════╝ ╚═════╝ ╚═╝╚═╝  ╚═══╝ ╚═════╝ ")
    print("="*80)
    print("🔮 EMBEDDING COMPLETED")
    print("="*80)
    print(f"📁 INPUT FILE: {input_path}")
    print(f"📄 OUTPUT FILE: {output_path}")
    print(f"🔢 TOTAL EMBEDDINGS: {len(outputs):,}")
    print(f"🤖 PROVIDER: {args.provider}")
    print(f"🧠 MODEL: {model_name}")
    if not args.no_cache:
        print(f"💾 CACHE: Saved to {cache_path}")
    print("✅ EMBEDDING COMPLETED SUCCESSFULLY!")
    print("="*80 + "\n")
    
    logging.info("Wrote %d embedding records to %s", written, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())


