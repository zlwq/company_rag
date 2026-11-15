import re
from typing import Any


CITATION_RE = re.compile(r"\[S(\d+)\]")


def _format_location(location: dict[str, Any] | None) -> str:
    location = location or {}

    if location.get("page") is not None:
        return f"第{location['page']}页"
    if location.get("slide") is not None:
        return f"第{location['slide']}页PPT"
    if location.get("sheet"):
        parts = [f"工作表 {location['sheet']}"]
        if location.get("cell_range"):
            parts.append(f"单元格 {location['cell_range']}")
        elif location.get("row") is not None:
            parts.append(f"第{location['row']}行")
        return "，".join(parts)
    if location.get("table") is not None:
        parts = [f"表格{location['table']}"]
        if location.get("row") is not None:
            parts.append(f"第{location['row']}行")
        return "，".join(parts)
    if location.get("paragraph") is not None:
        parts = [f"第{location['paragraph']}段"]
        if location.get("section"):
            parts.append(f"章节《{location['section']}》")
        return "，".join(parts)
    if location.get("line_start") is not None:
        start = location["line_start"]
        end = location.get("line_end", start)
        return f"第{start}-{end}行" if end != start else f"第{start}行"

    return "位置未标注"


def prepare_sources(hits: list[dict[str, Any]]) -> list[dict[str, Any]]:
    sources = []

    for index, hit in enumerate(hits, start=1):
        metadata = hit["metadata"]
        sources.append(
            {
                "citation_id": f"S{index}",
                "point_id": hit["point_id"],
                "score": hit["score"],
                "content": hit["content"],
                "document_id": metadata.get("document_id"),
                "file_hash": metadata.get("file_hash"),
                "file_name": metadata.get("file_name"),
                "file_type": metadata.get("file_type"),
                "source_path": metadata.get("source_path"),
                "project_name": metadata.get("project_name"),
                "document_type": metadata.get("document_type"),
                "version": metadata.get("version"),
                "document_date": metadata.get("document_date"),
                "source_created_at": metadata.get("source_created_at"),
                "source_modified_at": metadata.get("source_modified_at"),
                "keywords": metadata.get("keywords") or [],
                "summary": metadata.get("summary"),
                "location": metadata.get("location") or {},
                "location_text": _format_location(metadata.get("location")),
                "matched_queries": hit.get("matched_queries") or [],
            }
        )

    return sources


def validate_citations(answer: str, sources: list[dict[str, Any]]) -> list[str]:
    valid_ids = {source["citation_id"] for source in sources}
    cited = []

    for number in CITATION_RE.findall(answer):
        citation_id = f"S{number}"
        if citation_id not in valid_ids:
            raise ValueError(f"模型输出了不存在的引用: [{citation_id}]")
        if citation_id not in cited:
            cited.append(citation_id)

    return cited


def cited_sources(answer: str, sources: list[dict[str, Any]]) -> list[dict[str, Any]]:
    cited_ids = set(validate_citations(answer, sources))
    return [source for source in sources if source["citation_id"] in cited_ids]


def format_source_list(sources: list[dict[str, Any]]) -> str:
    lines = []

    for source in sources:
        extras = []
        if source.get("version"):
            extras.append(f"版本 {source['version']}")
        if source.get("document_date"):
            extras.append(f"文档日期 {source['document_date']}")
        elif source.get("source_created_at"):
            extras.append(f"创建时间 {source['source_created_at']}")
        elif source.get("source_modified_at"):
            extras.append(f"文件修改时间 {source['source_modified_at']}")

        if source.get("source_path"):
            extras.append(f"路径 {source['source_path']}")
        if source.get("document_id"):
            extras.append(f"document_id {str(source['document_id'])[:12]}")

        suffix = f"；{'；'.join(extras)}" if extras else ""
        lines.append(
            f"[{source['citation_id']}] 《{source.get('file_name') or '未知文件'}》"
            f"，{source['location_text']}{suffix}"
        )

    return "\n".join(lines)
