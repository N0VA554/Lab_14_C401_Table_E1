#!/usr/bin/env python3
"""
Document Chunker for RAG Pipeline.

Đọc tất cả file .md trong folder data/docs và chia thành các chunks
theo cấu trúc Markdown (heading-aware) với sliding-window overlap.

Output: data/chunks.jsonl
Mỗi dòng là một JSON object chứa:
  - chunk_id   : "doc_stem::chunk_###"
  - doc_id     : tên file (không đuôi)
  - heading    : heading cấp cao nhất của đoạn (nếu có)
  - text       : nội dung chunk
  - char_start : vị trí bắt đầu trong tài liệu gốc
  - char_end   : vị trí kết thúc
  - metadata   : { source_file, chunk_index, total_chunks, strategy }
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

# ─── CONFIG ──────────────────────────────────────────────────────────────────
DOCS_DIR    = Path(__file__).parent / "docs"
OUTPUT_FILE = Path(__file__).parent / "chunks.jsonl"

CHUNK_SIZE    = 500   # ký tự tối đa mỗi chunk
CHUNK_OVERLAP = 100   # ký tự overlap giữa 2 chunk liên tiếp

# Regex nhận diện heading Markdown (# / ## / ### ...)
HEADING_RE = re.compile(r"^(#{1,6})\s+(.+)$", re.MULTILINE)
# ─────────────────────────────────────────────────────────────────────────────


def split_by_headings(text: str) -> List[Tuple[str, str, int]]:
    """
    Chia văn bản thành các section theo heading Markdown.

    Trả về list[(heading_title, section_text, char_start)].
    Phần đầu tài liệu (trước heading đầu tiên) sẽ dùng heading="" 
    """
    sections: List[Tuple[str, str, int]] = []
    matches = list(HEADING_RE.finditer(text))

    if not matches:
        # Không có heading → cả file là 1 section
        return [("", text, 0)]

    # Phần trước heading đầu tiên
    if matches[0].start() > 0:
        sections.append(("", text[: matches[0].start()], 0))

    for idx, m in enumerate(matches):
        heading_title = m.group(2).strip()
        start = m.start()
        end   = matches[idx + 1].start() if idx + 1 < len(matches) else len(text)
        sections.append((heading_title, text[start:end], start))

    return sections


def sliding_window_chunks(
    text: str,
    chunk_size: int = CHUNK_SIZE,
    overlap: int = CHUNK_OVERLAP,
) -> List[Tuple[str, int, int]]:
    """
    Chia đoạn text thành các chunk bằng sliding window.
    Ưu tiên cắt tại dòng trống hoặc cuối câu (. ! ?) để tránh cắt giữa chừng.

    Trả về list[(chunk_text, rel_start, rel_end)].
    """
    if len(text) <= chunk_size:
        return [(text, 0, len(text))]

    chunks: List[Tuple[str, int, int]] = []
    start = 0

    while start < len(text):
        end = min(start + chunk_size, len(text))

        if end < len(text):
            # Thử cắt tại dòng trống gần nhất
            blank = text.rfind("\n\n", start, end)
            if blank != -1 and blank > start + overlap:
                end = blank + 2
            else:
                # Thử cắt tại cuối câu
                for sep in (". ", "! ", "? ", "\n"):
                    pos = text.rfind(sep, start + overlap, end)
                    if pos != -1:
                        end = pos + len(sep)
                        break

        chunk = text[start:end].strip()
        if chunk:
            chunks.append((chunk, start, end))

        next_start = end - overlap
        if next_start <= start:          # tránh vòng lặp vô hạn
            next_start = end
        start = next_start

    return chunks


def chunk_document(doc_stem: str, text: str) -> List[Dict]:
    """
    Chunk một tài liệu hoàn chỉnh.
    Chiến lược: chia theo heading → sliding window trong mỗi section.
    """
    all_chunks: List[Dict] = []
    sections = split_by_headings(text)

    global_idx = 0
    for heading, section_text, section_start in sections:
        sub_chunks = sliding_window_chunks(section_text)

        for chunk_text, rel_start, rel_end in sub_chunks:
            abs_start = section_start + rel_start
            abs_end   = section_start + rel_end

            chunk_id = f"{doc_stem}::chunk_{global_idx:03d}"
            all_chunks.append({
                "chunk_id"  : chunk_id,
                "doc_id"    : doc_stem,
                "heading"   : heading,
                "text"      : chunk_text,
                "char_start": abs_start,
                "char_end"  : abs_end,
                "metadata"  : {
                    "source_file" : f"{doc_stem}.md",
                    "chunk_index" : global_idx,
                    "strategy"    : "heading_aware_sliding_window",
                },
            })
            global_idx += 1

    # Cập nhật total_chunks sau khi biết tổng số
    for chunk in all_chunks:
        chunk["metadata"]["total_chunks"] = global_idx

    return all_chunks


def process_all_docs() -> List[Dict]:
    """Đọc tất cả .md trong DOCS_DIR và trả về toàn bộ chunks."""
    if not DOCS_DIR.exists():
        print(f"❌  Folder không tồn tại: {DOCS_DIR}")
        return []

    all_chunks: List[Dict] = []

    md_files = sorted(DOCS_DIR.glob("*.md"))
    if not md_files:
        print(f"⚠️   Không tìm thấy file .md trong {DOCS_DIR}")
        return []

    for md_file in md_files:
        print(f"📄  Đang xử lý: {md_file.name}")
        try:
            text = md_file.read_text(encoding="utf-8")
        except Exception as e:
            print(f"    ❌ Lỗi đọc file: {e}")
            continue

        doc_chunks = chunk_document(md_file.stem, text)
        all_chunks.extend(doc_chunks)

        print(f"    ✅ {len(doc_chunks)} chunks "
              f"| size ~{CHUNK_SIZE} chars | overlap {CHUNK_OVERLAP} chars")

    return all_chunks


def save_chunks(chunks: List[Dict]) -> None:
    """Lưu chunks ra JSONL."""
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        for chunk in chunks:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")
    print(f"\n💾  Đã lưu {len(chunks)} chunks → {OUTPUT_FILE}")


def print_summary(chunks: List[Dict]) -> None:
    """In thống kê nhanh."""
    if not chunks:
        return

    from collections import Counter
    doc_counts = Counter(c["doc_id"] for c in chunks)
    lengths    = [len(c["text"]) for c in chunks]

    print("\n" + "=" * 60)
    print("📊  CHUNKING SUMMARY")
    print("=" * 60)
    print(f"  Tổng số chunks : {len(chunks)}")
    print(f"  Tổng số docs   : {len(doc_counts)}")
    print(f"  Chunk size (ký tự): min={min(lengths)} / avg={int(sum(lengths)/len(lengths))} / max={max(lengths)}")
    print()
    print("  Phân bố theo tài liệu:")
    for doc, cnt in sorted(doc_counts.items()):
        print(f"    {doc}: {cnt} chunks")
    print("=" * 60)

    # Hiển thị 2 chunk mẫu
    print("\n🔍  Mẫu chunk đầu tiên:")
    sample = chunks[0]
    print(f"  chunk_id : {sample['chunk_id']}")
    print(f"  heading  : {sample['heading'] or '(preamble)'}")
    print(f"  chars    : {sample['char_start']} → {sample['char_end']}")
    print(f"  text     : {sample['text'][:200].replace(chr(10), ' ')} ...")


def main():
    print("🚀  Document Chunker — data/docs → data/chunks.jsonl")
    print(f"    CHUNK_SIZE={CHUNK_SIZE}  OVERLAP={CHUNK_OVERLAP}\n")

    chunks = process_all_docs()
    if not chunks:
        print("❌  Không có chunks nào được tạo.")
        return

    save_chunks(chunks)
    print_summary(chunks)
    print("\n✅  Hoàn tất! Dùng chunks.jsonl cho RAG pipeline & Retrieval Eval.")


if __name__ == "__main__":
    main()
