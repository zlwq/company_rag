"""打印 Word 文档中的表格；每张表的第一行视为表头。"""

import argparse
import shutil
import subprocess
import tempfile
from pathlib import Path

from docx import Document


def read_docx(path: Path):
    return Document(path)


def convert_doc(path: Path, output_dir: Path) -> Path:
    soffice = shutil.which("soffice") or shutil.which("libreoffice")
    if not soffice:
        raise RuntimeError("读取 .doc 需要安装 LibreOffice（命令 soffice）。")

    profile = (output_dir / "lo-profile").as_uri()
    result = subprocess.run(
        [soffice, "-env:UserInstallation=" + profile, "--headless",
         "--convert-to", "docx", "--outdir", str(output_dir), str(path)],
        capture_output=True, text=True,
    )
    converted = output_dir / (path.stem + ".docx")
    if result.returncode != 0 or not converted.exists():
        raise RuntimeError(f".doc 转换失败：{result.stderr or result.stdout}")
    return converted


def cell_text(cell) -> str:
    return " / ".join(line.strip() for line in cell.text.splitlines() if line.strip())


def print_tables(doc) -> None:
    if not doc.tables:
        print("没有找到表格。")
        return

    for index, table in enumerate(doc.tables, start=1):
        if index > 1:
            print("—————")
        print(f"表格 {index}")
        if not table.rows:
            print("（空表格）")
            continue
        print("表头：" + " | ".join(cell_text(c) for c in table.rows[0].cells))
        for row in table.rows[1:]:
            print("内容：" + " | ".join(cell_text(c) for c in row.cells))


def main() -> None:
    parser = argparse.ArgumentParser(description="输出 Word 文档中的表头和表格内容")
    parser.add_argument("path", type=Path, help=".doc 或 .docx 文件路径")
    args = parser.parse_args()

    path = args.path.expanduser().resolve()
    if not path.is_file() or path.suffix.lower() not in {".doc", ".docx"}:
        parser.error("请提供存在的 .doc 或 .docx 文件路径")

    try:
        if path.suffix.lower() == ".docx":
            print_tables(read_docx(path))
        else:
            with tempfile.TemporaryDirectory() as tmp:
                print_tables(read_docx(convert_doc(path, Path(tmp))))
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"错误：{exc}\n")


if __name__ == "__main__":
    main()
