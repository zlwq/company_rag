import hashlib
import json
import uuid
from typing import Any


def _hash_text(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _split_text(text: str, max_chars: int = 2200, overlap_chars: int = 150) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    chunks = []
    start = 0
    length = len(text)

    while start < length:
        end = min(start + max_chars, length)

        if end < length:
            search_start = start + int(max_chars * 0.6)
            best_cut = -1
            for sep in ["\n", "。", "！", "？", "；", ";", "."]:
                pos = text.rfind(sep, search_start, end)
                if pos > best_cut:
                    best_cut = pos + len(sep)
            if best_cut > start:
                end = best_cut

        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)

        if end >= length:
            break

        next_start = max(0, end - overlap_chars)
        if next_start <= start:
            next_start = end
        start = next_start

    return chunks


def build_chunks(
    parsed_document: dict[str, Any],
    document_id: str,
    file_hash: str,
    document_metadata: dict[str, Any] | None = None,
    max_chars: int = 2200,
    overlap_chars: int = 150,
) -> list[dict[str, Any]]:
    chunks = []
    document_metadata = document_metadata or {}

    for unit in parsed_document["units"]:
        # Word 表格必须完整保存在同一个 chunk，不能按字符数切开。
        if parsed_document["file_type"] == "docx" and unit["source_key"].startswith("table:"):
            parts = [unit["content"]]
        else:
            parts = _split_text(
                unit["content"],
                max_chars=max_chars,
                overlap_chars=overlap_chars,
            )

        for part_index, content in enumerate(parts, start=1):
            source_key = unit["source_key"]
            chunk_key = f"{source_key}:part:{part_index}"
            chunk_id = str(uuid.uuid5(uuid.NAMESPACE_URL, f"{document_id}:{chunk_key}"))

            hash_payload = {
                "content": content,
                "location": unit["location"],
                "chunk_key": chunk_key,
            }
            chunk_hash = _hash_text(
                json.dumps(hash_payload, ensure_ascii=False, sort_keys=True)
            )

            chunks.append(
                {
                    **document_metadata,
                    "chunk_id": chunk_id,
                    "chunk_key": chunk_key,
                    "chunk_hash": chunk_hash,
                    "document_id": document_id,
                    "file_hash": file_hash,
                    "file_name": parsed_document["file_name"],
                    "file_type": parsed_document["file_type"],
                    "source_path": parsed_document["source_path"],
                    "file_size": parsed_document.get("file_size"),
                    "source_modified_at": parsed_document.get("source_modified_at"),
                    "source_created_at": parsed_document.get("source_created_at"),
                    "content": content,
                    "location": unit["location"],
                }
            )

    return chunks