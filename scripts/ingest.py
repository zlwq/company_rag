import argparse
import hashlib
import sys
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
load_dotenv(ROOT / ".env")

from app.services.chunker import build_chunks
from app.services.embedder import Embedder
from app.services.metadata_extractor import MetadataExtractor
from app.services.parser import SUPPORTED_EXTENSIONS, parse_file
from app.services.qdrant_store import QdrantStore


def sha256_file(file_path: Path) -> str:
    digest = hashlib.sha256()
    with file_path.open("rb") as f:
        for block in iter(lambda: f.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def make_document_id(file_path: Path) -> str:
    normalized = str(file_path.expanduser().resolve())
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def collect_files(path: Path) -> list[Path]:
    if path.is_file():
        return [path] if path.suffix.lower() in SUPPORTED_EXTENSIONS else []

    if path.is_dir():
        return sorted(
            p
            for p in path.rglob("*")
            if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
        )

    raise FileNotFoundError(f"路径不存在: {path}")


def build_document_metadata(
    llm_metadata: dict,
    project_name_override: str | None,
    file_path: Path,
) -> dict:
    metadata = dict(llm_metadata)

    if project_name_override:
        metadata["project_name"] = project_name_override

    metadata["source_folder"] = file_path.parent.name
    return metadata


def ingest_file(
    file_path: Path,
    store: QdrantStore,
    project_name: str | None,
    max_chars: int,
    overlap_chars: int,
    use_llm_metadata: bool,
    force: bool = False,
) -> None:
    file_path = file_path.expanduser().resolve()
    file_hash = sha256_file(file_path)
    document_id = make_document_id(file_path)

    old_file_hash = store.get_document_file_hash(document_id)
    if old_file_hash == file_hash and not force:
        print(f"[跳过] 未修改: {file_path}")
        return

    if old_file_hash == file_hash and force:
        print(f"[强制刷新] {file_path}")

    print(f"[解析] {file_path}")
    parsed = parse_file(file_path)

    metadata_extractor = MetadataExtractor()
    if use_llm_metadata:
        print(f"[metadata] DeepSeek: {file_path.name}")
        llm_metadata = metadata_extractor.extract(parsed)
    else:
        llm_metadata = metadata_extractor.empty_metadata()

    document_metadata = build_document_metadata(
        llm_metadata=llm_metadata,
        project_name_override=project_name,
        file_path=file_path,
    )

    chunks = build_chunks(
        parsed_document=parsed,
        document_id=document_id,
        file_hash=file_hash,
        document_metadata=document_metadata,
        max_chars=max_chars,
        overlap_chars=overlap_chars,
    )

    old_points = store.get_document_points(document_id)
    new_by_id = {chunk["chunk_id"]: chunk for chunk in chunks}

    changed = []
    unchanged_ids = []

    for chunk_id, chunk in new_by_id.items():
        old_payload = old_points.get(chunk_id)
        if old_payload and old_payload.get("chunk_hash") == chunk["chunk_hash"]:
            unchanged_ids.append(chunk_id)
        else:
            changed.append(chunk)

    deleted_ids = [point_id for point_id in old_points if point_id not in new_by_id]

    if changed:
        embedder = Embedder()
        store.ensure_collection(embedder.dimension)
        vectors = embedder.embed(chunk["content"] for chunk in changed)
        store.upsert_chunks(changed, vectors)
    elif not store.collection_exists() and chunks:
        embedder = Embedder()
        store.ensure_collection(embedder.dimension)
        vectors = embedder.embed(chunk["content"] for chunk in chunks)
        store.upsert_chunks(chunks, vectors)
        changed = chunks
        unchanged_ids = []

    store.delete_points(deleted_ids)

    if unchanged_ids:
        store.set_payload(
            unchanged_ids,
            {
                **document_metadata,
                "file_hash": file_hash,
                "file_size": parsed.get("file_size"),
                "source_modified_at": parsed.get("source_modified_at"),
                "source_created_at": parsed.get("source_created_at"),
            },
        )

    print(
        f"[完成] {file_path.name} | 总chunk={len(chunks)} "
        f"| 新增/更新={len(changed)} | 未变化={len(unchanged_ids)} | 删除={len(deleted_ids)}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="公司文档增量入库到 Qdrant")
    parser.add_argument("path", help="单个文件或目录")
    parser.add_argument("--project", default=None, help="项目名称；显式指定时优先级最高")
    parser.add_argument("--collection", default=None, help="Qdrant collection 名称")
    parser.add_argument("--max-chars", type=int, default=2200, help="单个 chunk 最大字符数")
    parser.add_argument("--overlap-chars", type=int, default=150, help="超长内容切分时的重叠字符数")
    parser.add_argument(
        "--no-llm-metadata",
        action="store_true",
        help="不调用 DeepSeek 生成文档级 metadata",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="即使文件内容未变化，也重新解析并刷新 metadata",
    )
    args = parser.parse_args()

    path = Path(args.path).expanduser()
    files = collect_files(path)
    if not files:
        print("没有找到支持的文档。")
        return

    store = QdrantStore(collection_name=args.collection)

    for file_path in files:
        try:
            ingest_file(
                file_path=file_path,
                store=store,
                project_name=args.project,
                max_chars=args.max_chars,
                overlap_chars=args.overlap_chars,
                use_llm_metadata=not args.no_llm_metadata,
                force=args.force,
            )
        except Exception as exc:
            print(f"[失败] {file_path}: {exc}")


if __name__ == "__main__":
    main()