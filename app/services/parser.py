from datetime import datetime, timezone
from pathlib import Path
import subprocess
from typing import Any

import fitz
from docx import Document
from openpyxl import load_workbook
from openpyxl.utils import get_column_letter
from pptx import Presentation


SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".txt", ".md", ".xlsx", ".pptx"}


def _clean_text(text: str) -> str:
    lines = [
        line.rstrip()
        for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")
    ]
    return "\n".join(lines).strip()


def _iso_time(timestamp: float) -> str:
    return datetime.fromtimestamp(timestamp, tz=timezone.utc).isoformat()




def _birth_time(file_path: Path, stat_result: Any) -> str | None:
    if hasattr(stat_result, "st_birthtime"):
        return _iso_time(stat_result.st_birthtime)

    # Linux 的 Python stat_result 通常没有 st_birthtime；GNU stat 的 %W
    # 在底层文件系统支持 birth time 时可以拿到 Unix 时间戳。
    try:
        result = subprocess.run(
            ["stat", "-c", "%W", str(file_path)],
            capture_output=True,
            text=True,
            check=True,
        )
        timestamp = int(result.stdout.strip() or "0")
        if timestamp > 0:
            return _iso_time(timestamp)
    except (OSError, ValueError, subprocess.CalledProcessError):
        pass

    return None

def _unit(content: str, source_key: str, location: dict[str, Any]) -> dict[str, Any]:
    return {
        "content": _clean_text(content),
        "source_key": source_key,
        "location": location,
    }


def parse_pdf(file_path: Path) -> list[dict[str, Any]]:
    units = []
    with fitz.open(file_path) as doc:
        for page_index, page in enumerate(doc):
            text = _clean_text(page.get_text("text", sort=True))
            if not text:
                continue
            page_number = page_index + 1
            units.append(
                _unit(
                    content=text,
                    source_key=f"page:{page_number}",
                    location={"page": page_number},
                )
            )
    return units


def parse_docx(file_path: Path) -> list[dict[str, Any]]:
    doc = Document(file_path)
    units = []
    current_section = None

    for paragraph_index, paragraph in enumerate(doc.paragraphs, start=1):
        text = _clean_text(paragraph.text)
        if not text:
            continue

        style_name = paragraph.style.name if paragraph.style else ""
        if style_name.lower().startswith("heading"):
            current_section = text

        units.append(
            _unit(
                content=text,
                source_key=f"paragraph:{paragraph_index}",
                location={
                    "paragraph": paragraph_index,
                    "section": current_section,
                },
            )
        )

    def cell_text(cell) -> str:
        return " / ".join(line.strip() for line in cell.text.splitlines() if line.strip())

    for table_index, table in enumerate(doc.tables, start=1):
        lines = [f"表格 {table_index}"]
        if table.rows:
            lines.append("表头：" + " | ".join(cell_text(c) for c in table.rows[0].cells))
            for row in table.rows[1:]:
                lines.append("内容：" + " | ".join(cell_text(c) for c in row.cells))
        else:
            lines.append("（空表格）")

        units.append(
            _unit(
                content="\n".join(lines),
                source_key=f"table:{table_index}",
                location={"table": table_index},
            )
        )

    return units


def parse_txt(file_path: Path) -> list[dict[str, Any]]:
    text = file_path.read_text(encoding="utf-8-sig", errors="replace")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    units = []
    buffer = []
    start_line = None

    def flush(end_line: int) -> None:
        nonlocal buffer, start_line
        if not buffer or start_line is None:
            buffer = []
            start_line = None
            return

        content = _clean_text("\n".join(buffer))
        if content:
            units.append(
                _unit(
                    content=content,
                    source_key=f"lines:{start_line}-{end_line}",
                    location={
                        "line_start": start_line,
                        "line_end": end_line,
                    },
                )
            )

        buffer = []
        start_line = None

    for line_number, line in enumerate(lines, start=1):
        if line.strip():
            if start_line is None:
                start_line = line_number
            buffer.append(line)
        else:
            flush(line_number - 1)

    flush(len(lines))
    return units


def parse_xlsx(file_path: Path) -> list[dict[str, Any]]:
    workbook = load_workbook(file_path, read_only=True, data_only=True)
    units = []

    try:
        for sheet in workbook.worksheets:
            for row_index, row in enumerate(sheet.iter_rows(), start=1):
                non_empty = []
                for col_index, cell in enumerate(row, start=1):
                    if cell.value is None or str(cell.value).strip() == "":
                        continue
                    non_empty.append((col_index, str(cell.value)))

                if not non_empty:
                    continue

                first_col = non_empty[0][0]
                last_col = non_empty[-1][0]
                cell_range = (
                    f"{get_column_letter(first_col)}{row_index}:"
                    f"{get_column_letter(last_col)}{row_index}"
                )
                text = " | ".join(
                    f"{get_column_letter(col)}={value}" for col, value in non_empty
                )

                units.append(
                    _unit(
                        content=text,
                        source_key=f"sheet:{sheet.title}:row:{row_index}",
                        location={
                            "sheet": sheet.title,
                            "row": row_index,
                            "cell_range": cell_range,
                        },
                    )
                )
    finally:
        workbook.close()

    return units


def parse_pptx(file_path: Path) -> list[dict[str, Any]]:
    presentation = Presentation(file_path)
    units = []

    for slide_index, slide in enumerate(presentation.slides, start=1):
        parts = []

        for shape in slide.shapes:
            if getattr(shape, "has_text_frame", False):
                text = _clean_text(shape.text)
                if text:
                    parts.append(text)

            if getattr(shape, "has_table", False):
                for row in shape.table.rows:
                    values = [_clean_text(cell.text) for cell in row.cells]
                    if any(values):
                        parts.append(" | ".join(values))

        content = _clean_text("\n".join(parts))
        if not content:
            continue

        units.append(
            _unit(
                content=content,
                source_key=f"slide:{slide_index}",
                location={"slide": slide_index},
            )
        )

    return units


def parse_file(file_path: str | Path) -> dict[str, Any]:
    path = Path(file_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"文件不存在: {path}")

    extension = path.suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"暂不支持的文件类型: {extension}")

    parser_map = {
        ".pdf": parse_pdf,
        ".docx": parse_docx,
        ".txt": parse_txt,
        ".md": parse_txt,
        ".xlsx": parse_xlsx,
        ".pptx": parse_pptx,
    }

    stat = path.stat()
    source_created_at = _birth_time(path, stat)

    return {
        "file_name": path.name,
        "file_type": extension.lstrip("."),
        "source_path": str(path),
        "file_size": stat.st_size,
        "source_modified_at": _iso_time(stat.st_mtime),
        "source_created_at": source_created_at,
        "units": parser_map[extension](path),
    }