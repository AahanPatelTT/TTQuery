#!/usr/bin/env python3
"""
Chunking step for the Synapse pipeline (Parse -> Chunk -> Embed -> Retrieve -> Generate)

This script takes the normalized JSONL produced by the parse step and produces
token-targeted chunks (default ~300 tokens) with overlap for better recall. It
is heading-aware for Markdown and uses a semantic fallback for other formats.

Design choices and rationale:
- Use MarkdownHeaderTextSplitter to preserve section semantics on Markdown sources.
- Use RecursiveCharacterTextSplitter with a token-based length function so chunk
  sizes reflect the embedding model's tokenization characteristics.
- Preserve and propagate provenance metadata so downstream retrieval can provide
  precise citations. We add token counts and chunk indices for traceability.

Alternate approaches:
- Semantic text splitter (e.g., using embeddings to split on semantic boundaries).
  Higher quality but slower; suitable later if chunk cohesion matters.
- Code-aware chunkers (tree-sitter) for repositories. Add when you ingest code.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import dataclass
from typing import Callable, Dict, Iterable, Iterator, List, Optional, Tuple
import re


# -------------------------------
# Utilities
# -------------------------------


def compute_sha1(text: str) -> str:
    import hashlib

    h = hashlib.sha1()
    h.update(text.encode("utf-8", errors="ignore"))
    return h.hexdigest()


def compute_document_id(source_path: str) -> str:
    abs_path = os.path.abspath(source_path)
    return compute_sha1(f"doc::{abs_path}")


def compute_chunk_id(document_id: str, locator: str, content: str) -> str:
    content_hash = compute_sha1(content)[:16]
    return compute_sha1(f"chunk::{document_id}::{locator}::{content_hash}")


# -------------------------------
# Tokenization support
# -------------------------------


def build_token_len_fn(encoding_name: str = "cl100k_base") -> Callable[[str], int]:
    """Return a function that counts tokens. Falls back to char-based heuristic.

    We default to OpenAI's cl100k_base which approximates many modern LLM BPEs.
    """

    try:
        import tiktoken  # type: ignore

        enc = tiktoken.get_encoding(encoding_name)

        def _length(text: str) -> int:
            return len(enc.encode(text))

        return _length
    except Exception:

        def _length(text: str) -> int:
            # Heuristic: ~4 chars per token
            return max(1, len(text) // 4)

        return _length


# -------------------------------
# Chunking primitives
# -------------------------------


def split_markdown_heading_aware(
    markdown_text: str,
    token_len: Callable[[str], int],
    target_tokens: int,
    overlap_tokens: int,
) -> List[Tuple[str, List[str]]]:
    """Split Markdown by headings, then token-size chunks, returning (text, heading_path).

    We extract heading metadata from the Markdown splitter and propagate it.
    """

    try:
        from langchain_text_splitters import (
            MarkdownHeaderTextSplitter,
            RecursiveCharacterTextSplitter,
        )
    except Exception as exc:  # pragma: no cover
        raise RuntimeError(
            "langchain-text-splitters is required. Install with 'pip install langchain-text-splitters'."
        ) from exc

    headers_to_split_on = [
        ("#", "H1"),
        ("##", "H2"),
        ("###", "H3"),
    ]
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=headers_to_split_on)
    docs = md_splitter.split_text(markdown_text)

    char_splitter = RecursiveCharacterTextSplitter(
        chunk_size=target_tokens,
        chunk_overlap=overlap_tokens,
        length_function=token_len,
        separators=["\n\n", "\n", " ", ""],
    )

    def _looks_like_md_table_header(header_line: str, next_line: str) -> bool:
        # Basic GFM detection: header with pipes followed by a separator line of dashes/colons and pipes
        if "|" not in header_line:
            return False
        sep_re = re.compile(r"^\s*\|?\s*:?-{3,}:?\s*(\|\s*:?-{3,}:?\s*)+\|?\s*$")
        return bool(sep_re.match(next_line))

    def _split_blocks_preserving_tables(text: str) -> List[Tuple[str, bool]]:
        """Return blocks as (block_text, is_table). Markdown tables are kept intact.

        Detection supports both pipe-wrapped and GFM tables without outer pipes.
        """
        if not text:
            return []
        lines = text.splitlines(True)  # keep line endings
        i = 0
        blocks: List[Tuple[str, bool]] = []
        while i < len(lines):
            line = lines[i]
            next_line = lines[i + 1] if i + 1 < len(lines) else ""
            if _looks_like_md_table_header(line, next_line):
                start = i
                i += 2
                # consume subsequent table rows (heuristic: lines containing at least one pipe)
                while i < len(lines):
                    ln = lines[i]
                    if ln.strip() == "":
                        break
                    if "|" not in ln:
                        break
                    i += 1
                blocks.append(("".join(lines[start:i]).rstrip("\n"), True))
            else:
                start = i
                i += 1
                while i < len(lines):
                    cur = lines[i]
                    nxt = lines[i + 1] if i + 1 < len(lines) else ""
                    if _looks_like_md_table_header(cur, nxt):
                        break
                    i += 1
                blocks.append(("".join(lines[start:i]).rstrip("\n"), False))
        return [b for b in blocks if b[0].strip()]

    results: List[Tuple[str, List[str]]] = []
    for d in docs:
        # LangChain returns a Document with metadata like {"H1": "Title", "H2": "Section"}
        meta = getattr(d, "metadata", {}) or {}
        heading_path = [meta[k] for k in ["H1", "H2", "H3"] if k in meta and meta[k]]

        # Split content into blocks where Markdown tables are atomic
        blocks = _split_blocks_preserving_tables(d.page_content)
        for block_text, is_table in blocks:
            if is_table:
                # Keep the entire table as one atomic chunk, even if it exceeds target size
                if block_text.strip():
                    results.append((block_text, heading_path))
            else:
                for chunk in char_splitter.split_text(block_text):
                    results.append((chunk, heading_path))

    # If nothing was produced (no headings), fallback to semantic splitter only
    if not results:
        char_only = RecursiveCharacterTextSplitter(
            chunk_size=target_tokens,
            chunk_overlap=overlap_tokens,
            length_function=token_len,
            separators=["\n\n", "\n", " ", ""],
        ).split_text(markdown_text)
        results = [(c, []) for c in char_only]

    return results


def sentence_spans(text: str) -> List[Tuple[int, int]]:
    """Return list of (start, end) character spans for sentences.

    Heuristic sentence splitter using regex. Designed to be dependency-free and
    robust for engineering prose. Falls back to paragraph lines if needed.
    """
    if not text:
        return []

    # Normalize line endings to help regex operate predictably
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")

    # Split into paragraphs first to avoid spanning across large blocks
    paragraphs = normalized.split("\n\n")
    spans: List[Tuple[int, int]] = []
    base = 0
    sentence_end_re = re.compile(r"([.!?])\s+(?=[A-Z0-9\[(])")

    for para in paragraphs:
        if not para.strip():
            base += len(para) + 2  # account for the two newlines
            continue
        start = 0
        last_end = 0
        for m in sentence_end_re.finditer(para):
            end = m.end(1)  # include the punctuation
            # Skip common abbreviations like e.g., i.e., or single-letter initials
            prefix = para[max(0, end - 4):end].lower()
            if prefix.endswith("e.g") or prefix.endswith("i.e"):
                continue
            # Skip patterns like "A." where A is capital and likely an initial (heuristic)
            if end >= 2 and para[end-2].isupper() and para[end-1] == ".":
                continue
            spans.append((base + start, base + end))
            start = m.end()
            last_end = end
        # Remaining tail
        if start < len(para):
            spans.append((base + start, base + len(para)))

        base += len(para) + 2

    # Fallback: if the heuristic produced extremely long spans or too few, split by lines
    if len(spans) <= 1 and len(normalized) > 0:
        spans = []
        base = 0
        for line in normalized.splitlines(True):  # keepends
            if line.strip():
                spans.append((base, base + len(line)))
            base += len(line)
    return spans


def smart_pack_text(
    text: str, token_len: Callable[[str], int], target_tokens: int, overlap_tokens: int
) -> List[Tuple[str, int, int]]:
    """Pack sentences into token-bounded chunks with overlap.

    Returns a list of tuples: (chunk_text, char_start, char_end), where char spans
    refer to offsets within the original text.
    """
    if not text.strip():
        return []

    spans = sentence_spans(text)
    # Merge extremely short sentences with neighbors to avoid fragmentation
    merged: List[Tuple[int, int]] = []
    min_tokens_per_sentence = max(1, target_tokens // 20)  # ~5% of target
    buffer_start = None
    buffer_end = None
    buffer_tokens = 0
    for (s, e) in spans:
        sent = text[s:e]
        t = token_len(sent)
        if buffer_start is None:
            buffer_start, buffer_end, buffer_tokens = s, e, t
        else:
            if buffer_tokens < min_tokens_per_sentence:
                buffer_end = e
                buffer_tokens += t
            else:
                merged.append((buffer_start, buffer_end))
                buffer_start, buffer_end, buffer_tokens = s, e, t
    if buffer_start is not None:
        merged.append((buffer_start, buffer_end))

    # Pack into chunks up to target_tokens with overlap by sentence spans
    chunks: List[Tuple[str, int, int]] = []
    window: List[Tuple[int, int]] = []
    window_tokens = 0

    def window_tokens_len(win: List[Tuple[int, int]]) -> int:
        if not win:
            return 0
        return token_len("".join(text[s:e] for s, e in win))

    for span in merged:
        s, e = span
        piece = text[s:e]
        piece_tokens = token_len(piece)
        if window and window_tokens + piece_tokens > target_tokens:
            # Emit current window
            start_w = window[0][0]
            end_w = window[-1][1]
            chunks.append((text[start_w:end_w], start_w, end_w))

            # Build overlap by retaining tail of current window
            overlap: List[Tuple[int, int]] = []
            overlap_tok = 0
            for rs, re in reversed(window):
                seg = text[rs:re]
                seg_t = token_len(seg)
                if overlap_tok + seg_t >= overlap_tokens and overlap:
                    break
                overlap.append((rs, re))
                overlap_tok += seg_t
            overlap.reverse()
            window = overlap[:]
            window_tokens = window_tokens_len(window)

        window.append((s, e))
        window_tokens += piece_tokens

    if window:
        start_w = window[0][0]
        end_w = window[-1][1]
        chunks.append((text[start_w:end_w], start_w, end_w))

    return chunks


def split_csv_rows(
    csv_text: str, token_len: Callable[[str], int], target_tokens: int, overlap_tokens: int
) -> List[str]:
    """Row-aware splitter for CSV text so we do not split across rows.

    Packs rows into chunks approximating target token sizes; applies overlap in rows.
    """

    lines = csv_text.splitlines()
    if not lines:
        return []

    header = lines[0]
    rows = lines[1:]
    chunks: List[str] = []
    current: List[str] = [header]
    current_tokens = token_len(header) + token_len("\n")

    for row in rows:
        row_tokens = token_len(row) + token_len("\n")
        if current_tokens + row_tokens > target_tokens and len(current) > 1:
            chunks.append("\n".join(current))
            # overlap by rows (approx proportional to tokens)
            overlap_rows = max(0, int(max(1, len(current) - 1) * (overlap_tokens / max(1, current_tokens))))
            if overlap_rows > 0:
                current = [header] + current[-overlap_rows:]
                current_tokens = sum(token_len(x) + token_len("\n") for x in current)
            else:
                current = [header]
                current_tokens = token_len(header) + token_len("\n")
        current.append(row)
        current_tokens += row_tokens

    if len(current) > 1:
        chunks.append("\n".join(current))
    return chunks


# -------------------------------
# I/O helpers
# -------------------------------


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
# Main driver
# -------------------------------


@dataclass
class ChunkingConfig:
    target_tokens: int = 300
    overlap_ratio: float = 0.12  # 12% overlap
    encoding_name: str = "cl100k_base"


def chunk_records(records: List[Dict[str, object]], cfg: ChunkingConfig) -> List[Dict[str, object]]:
    token_len = build_token_len_fn(cfg.encoding_name)
    overlap_tokens = max(0, int(cfg.target_tokens * cfg.overlap_ratio))

    # Group markdown by source to allow heading-aware splitting across the whole file
    md_groups: Dict[str, List[Dict[str, object]]] = {}
    # Group pptx by source to allow slide-order-aware processing and windowed chunks
    pptx_groups: Dict[str, List[Dict[str, object]]] = {}
    others: List[Dict[str, object]] = []
    for r in records:
        source_type = str(r.get("source_type", ""))
        if source_type == "md":
            src = str(r.get("source_path", ""))
            md_groups.setdefault(src, []).append(r)
        elif source_type == "pptx":
            src = str(r.get("source_path", ""))
            pptx_groups.setdefault(src, []).append(r)
        else:
            others.append(r)

    # Sort markdown groups by section_index if available
    for src, lst in md_groups.items():
        lst.sort(key=lambda x: int(x.get("metadata", {}).get("section_index", 0)))

    output_chunks: List[Dict[str, object]] = []

    # Process Markdown groups with heading-aware splitter
    for src, lst in md_groups.items():
        # Concatenate sections back into a single Markdown document
        concatenated = "\n\n".join(str(r.get("content", "")) for r in lst if r.get("content"))
        if not concatenated.strip():
            continue

        doc_id = compute_document_id(src)
        results = split_markdown_heading_aware(
            concatenated, token_len, cfg.target_tokens, overlap_tokens
        )
        for idx, (chunk_text, heading_path) in enumerate(results, start=1):
            locator = f"md-chunk:{idx}"
            chunk_id = compute_chunk_id(doc_id, locator, chunk_text)
            output_chunks.append(
                {
                    "id": chunk_id,
                    "document_id": doc_id,
                    "source_path": src,
                    "source_type": "md",
                    "content": chunk_text,
                    "metadata": {
                        "heading_path": heading_path,
                        "chunk_index": idx,
                        "content_format": "markdown",
                        "num_tokens": token_len(chunk_text),
                        "file_name": os.path.basename(src),
                    },
                }
            )

    # Process PPTX groups: keep each slide as an atomic chunk and also create windowed slide chunks
    PPTX_WINDOW: int = 2  # number of slides per window chunk
    PPTX_STRIDE: int = 1
    for src, lst in pptx_groups.items():
        # Sort by slide_number if available
        lst.sort(key=lambda x: int((x.get("metadata", {}) or {}).get("slide_number") or 0))
        doc_id = compute_document_id(src)
        # Emit single-slide chunks
        for idx, r in enumerate(lst, start=1):
            content = str(r.get("content", "") or "")
            if not content.strip():
                continue
            locator = f"pptx-slide:{idx}"
            chunk_id = compute_chunk_id(doc_id, locator, content)
            meta = dict(r.get("metadata", {}) or {})
            meta.update({
                "chunk_index": idx,
                "num_tokens": token_len(content),
                "content_format": meta.get("content_format") or "pptx_slide",
                "file_name": os.path.basename(src),
            })
            output_chunks.append(
                {
                    "id": chunk_id,
                    "document_id": doc_id,
                    "source_path": src,
                    "source_type": "pptx",
                    "content": content,
                    "metadata": meta,
                }
            )
        # Emit windowed multi-slide chunks for better section recall
        n = len(lst)
        if PPTX_WINDOW >= 2 and n >= 2:
            w_index = 1
            for start in range(0, n, PPTX_STRIDE):
                end = min(n, start + PPTX_WINDOW)
                if end - start < 2:
                    continue
                window = lst[start:end]
                contents = [str(r.get("content", "") or "") for r in window]
                merged = "\n\n---\n\n".join([c for c in contents if c.strip()])
                if not merged.strip():
                    continue
                locator = f"pptx-window:{window[0].get('metadata', {}).get('slide_number')}-{window[-1].get('metadata', {}).get('slide_number')}"
                chunk_id = compute_chunk_id(doc_id, locator, merged)
                meta0 = dict(window[0].get("metadata", {}) or {})
                metaN = dict(window[-1].get("metadata", {}) or {})
                new_meta = {
                    "chunk_index": int(meta0.get("slide_number") or start + 1),
                    "num_tokens": token_len(merged),
                    "content_format": "pptx_slide_window",
                    "file_name": os.path.basename(src),
                    "slide_range": [meta0.get("slide_number"), metaN.get("slide_number")],
                }
                output_chunks.append(
                    {
                        "id": chunk_id,
                        "document_id": doc_id,
                        "source_path": src,
                        "source_type": "pptx",
                        "content": merged,
                        "metadata": new_meta,
                    }
                )
                w_index += 1

    # Process other records one-by-one using smart or CSV-aware chunkers
    for r in others:
        content = str(r.get("content", "") or "")
        if not content.strip():
            continue
        src = str(r.get("source_path", ""))
        doc_id = str(r.get("document_id") or compute_document_id(src))
        source_type = str(r.get("source_type", ""))
        meta = dict(r.get("metadata", {}) or {})

        # Decide chunking strategy
        if meta.get("content_format") == "csv" or source_type == "csv":
            chunks = split_csv_rows(content, token_len, cfg.target_tokens, overlap_tokens)
        elif source_type == "pptx":
            # Already handled in pptx_groups to preserve order and add windowed chunks
            continue
        else:
            # Smart sentence-aware packing for coherence
            packed = smart_pack_text(content, token_len, cfg.target_tokens, overlap_tokens)
            chunks = [t for t, _, _ in packed]

        for idx, chunk_text in enumerate(chunks, start=1):
            locator = f"{source_type or 'doc'}-chunk:{idx}"
            chunk_id = compute_chunk_id(doc_id, locator, chunk_text)
            new_meta = dict(meta)
            # Enrich metadata for provenance and coherence
            new_meta.update({
                "chunk_index": idx,
                "num_tokens": token_len(chunk_text),
                # Add a small rolling fingerprint to help deduplicate near-duplicates later
                "content_sha1": compute_sha1(chunk_text)[:16],
            })
            output_chunks.append(
                {
                    "id": chunk_id,
                    "document_id": doc_id,
                    "source_path": src,
                    "source_type": source_type,
                    "content": chunk_text,
                    "metadata": new_meta,
                }
            )

    # Stable ordering for determinism
    output_chunks.sort(key=lambda c: (c["source_path"], c["metadata"].get("chunk_index", 0), c["id"]))
    return output_chunks


def main(argv: Optional[List[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Chunk parsed JSONL into token-targeted chunks")
    default_input = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "parsed.jsonl"))
    default_output = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "artifacts", "chunked.jsonl"))
    parser.add_argument("--input", type=str, default=default_input, help="Input parsed JSONL path")
    parser.add_argument("--output", type=str, default=default_output, help="Output chunked JSONL path")
    parser.add_argument("--target-tokens", type=int, default=300, help="Approx target tokens per chunk")
    parser.add_argument("--overlap", type=float, default=0.12, help="Overlap ratio between chunks (0-1)")
    parser.add_argument("--encoding", type=str, default="cl100k_base", help="Tokenizer encoding name")
    parser.add_argument("--verbose", action="store_true", help="Verbose logging")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)s %(message)s",
    )

    input_path = os.path.abspath(args.input)
    output_path = os.path.abspath(args.output)

    if not os.path.isfile(input_path):
        logging.error("Input file does not exist: %s", input_path)
        return 2

    logging.info("Reading parsed records from %s", input_path)
    records = list(read_jsonl(input_path))
    if not records:
        logging.warning("No input records found.")

    cfg = ChunkingConfig(
        target_tokens=max(50, int(args.target_tokens)),
        overlap_ratio=max(0.0, min(0.5, float(args.overlap))),
        encoding_name=str(args.encoding),
    )
    logging.info(
        "Chunking with target_tokens=%d, overlap=%.2f, encoding=%s",
        cfg.target_tokens,
        cfg.overlap_ratio,
        cfg.encoding_name,
    )

    chunks = chunk_records(records, cfg)
    written = write_jsonl(chunks, output_path)
    logging.info("Wrote %d chunks to %s", written, output_path)
    return 0


if __name__ == "__main__":
    sys.exit(main())



