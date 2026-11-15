import argparse
import json
from pathlib import Path
from typing import Any, Callable

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parents[2]
load_dotenv(ROOT / ".env")

from app.services.citations import cited_sources, format_source_list, prepare_sources
from app.services.deepseek import DeepSeekClient
from app.services.retriever import Retriever


class RAGService:
    def __init__(self) -> None:
        self.retriever = Retriever()
        self.deepseek = DeepSeekClient()

    def ask(
        self,
        question: str,
        top_k: int | None = None,
        project_name: str | None = None,
        document_type: str | None = None,
        version: str | list[str] | None = None,
        file_name: str | None = None,
        customer_name: str | None = None,
        file_type: str | None = None,
        project_stage: str | None = None,
        document_date: str | None = None,
        access_scope: str | None = None,
        is_active: bool | None = None,
        progress_callback: Callable[[str], None] | None = None,
    ) -> dict[str, Any]:
        del top_k  # 新架构中 chunk 数量由 finalize_documents 的 chunk_mode 决定。

        explicit_filters = {
            "project_name": project_name,
            "document_type": document_type,
            "version": version,
            "file_name": file_name,
            "customer_name": customer_name,
            "file_type": file_type,
            "project_stage": project_stage,
            "document_date": document_date,
            "access_scope": access_scope,
            "is_active": is_active,
        }

        if progress_callback:
            progress_callback("正在理解问题…")

        route = self.deepseek.route_query(question)
        retrieval_mode = route["retrieval_mode"]

        if retrieval_mode == "none":
            if progress_callback:
                progress_callback("正在生成回答…")
            answer = self.deepseek.answer_general(question)
            return {
                "answer": answer,
                "raw_answer": answer,
                "sources": [],
                "retrieved_count": 0,
                "documents": [],
                "duplicate_groups": [],
                "chunk_mode": "none",
                "detail_queries": [],
                "retrieval_route": route,
                "tool_trace": [],
                "hits": [],
            }

        session = self.retriever.new_session(explicit_filters=explicit_filters)
        agent_result = self.deepseek.select_documents(
            question=question,
            execute_tool=session.execute,
            forced_chunk_mode=(
                "metadata_only" if retrieval_mode == "metadata_only" else None
            ),
            progress_callback=progress_callback,
        )

        if not session.finalized:
            raise RuntimeError("文档检索流程没有完成 finalize_documents。")

        documents = list(session.documents)
        duplicate_groups = self.retriever.duplicate_groups(documents)

        if not documents:
            return {
                "answer": "资料库中没有找到符合用户条件的相应文件。",
                "raw_answer": None,
                "sources": [],
                "retrieved_count": 0,
                "documents": [],
                "duplicate_groups": [],
                "chunk_mode": session.chunk_mode,
                "detail_queries": session.detail_queries,
                "retrieval_route": route,
                "tool_trace": agent_result.get("tool_trace") or session.trace,
                "hits": [],
            }

        if session.chunk_mode == "metadata_only":
            if progress_callback:
                progress_callback("正在整理文件信息…")
            answer = self.deepseek.answer_metadata(
                question=question,
                documents=documents,
                duplicate_groups=duplicate_groups,
            )
            return {
                "answer": answer,
                "raw_answer": answer,
                "sources": [],
                "retrieved_count": 0,
                "documents": documents,
                "duplicate_groups": duplicate_groups,
                "chunk_mode": session.chunk_mode,
                "detail_queries": [],
                "retrieval_route": route,
                "tool_trace": agent_result.get("tool_trace") or session.trace,
                "hits": [],
            }

        if progress_callback:
            progress_callback(
                "正在读取完整文档…"
                if session.chunk_mode == "all"
                else "正在检索相关内容…"
            )

        hits = self.retriever.load_chunks(
            documents=documents,
            mode=session.chunk_mode,
            detail_queries=session.detail_queries,
        )

        if not hits:
            return {
                "answer": "已经找到符合条件的文件，但这些文件中没有可用于回答的正文内容。",
                "raw_answer": None,
                "sources": [],
                "retrieved_count": 0,
                "documents": documents,
                "duplicate_groups": duplicate_groups,
                "chunk_mode": session.chunk_mode,
                "detail_queries": session.detail_queries,
                "retrieval_route": route,
                "tool_trace": agent_result.get("tool_trace") or session.trace,
                "hits": [],
            }

        sources = prepare_sources(hits)
        if progress_callback:
            progress_callback("正在组织答案…")
        answer = self.deepseek.answer(
            question=question,
            sources=sources,
            documents=documents,
            duplicate_groups=duplicate_groups,
        )
        used_sources = cited_sources(answer, sources)

        if not used_sources:
            raise ValueError("DeepSeek 返回了无引用答案，已拒绝输出。")

        source_text = format_source_list(used_sources)
        final_answer = f"{answer}\n\n引用来源：\n{source_text}"

        return {
            "answer": final_answer,
            "raw_answer": answer,
            "sources": used_sources,
            "retrieved_count": len(hits),
            "documents": documents,
            "duplicate_groups": duplicate_groups,
            "chunk_mode": session.chunk_mode,
            "detail_queries": session.detail_queries,
            "retrieval_route": route,
            "tool_trace": agent_result.get("tool_trace") or session.trace,
            "hits": hits,
        }


def _document_debug_view(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    result = []
    for document in documents:
        metadata = document.get("metadata") or {}
        result.append(
            {
                "document_id": document.get("document_id"),
                "file_name": metadata.get("file_name"),
                "version": metadata.get("version"),
                "document_type": metadata.get("document_type"),
                "document_date": metadata.get("document_date"),
                "source_created_at": metadata.get("source_created_at"),
                "source_modified_at": metadata.get("source_modified_at"),
                "source_path": metadata.get("source_path"),
                "file_hash": metadata.get("file_hash"),
            }
        )
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="公司知识库 RAG 查询")
    parser.add_argument("question", help="用户问题")
    parser.add_argument("--top-k", type=int, default=None)
    parser.add_argument("--project", default=None)
    parser.add_argument("--document-type", default=None)
    parser.add_argument(
        "--version",
        action="append",
        default=None,
        help="显式精确版本过滤，可重复使用",
    )
    parser.add_argument("--file-name", default=None)
    parser.add_argument("--customer", default=None)
    parser.add_argument("--file-type", default=None)
    parser.add_argument("--project-stage", default=None)
    parser.add_argument("--document-date", default=None)
    parser.add_argument("--access-scope", default=None)
    parser.add_argument(
        "--active",
        action="store_true",
        help="仅在 payload 已有 is_active 字段时使用",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="显示 LLM 工具调用、最终文件集合和 chunk",
    )
    args = parser.parse_args()

    service = RAGService()
    result = service.ask(
        question=args.question,
        top_k=args.top_k,
        project_name=args.project,
        document_type=args.document_type,
        version=args.version,
        file_name=args.file_name,
        customer_name=args.customer,
        file_type=args.file_type,
        project_stage=args.project_stage,
        document_date=args.document_date,
        access_scope=args.access_scope,
        is_active=True if args.active else None,
    )

    if args.debug:
        print("\n[检索路由]")
        print(json.dumps(result.get("retrieval_route") or {}, ensure_ascii=False, indent=2))

        print("\n[LLM 工具调用轨迹]")
        print(json.dumps(result.get("tool_trace") or [], ensure_ascii=False, indent=2))

        print("\n[最终文件集合]")
        print(
            json.dumps(
                _document_debug_view(result.get("documents") or []),
                ensure_ascii=False,
                indent=2,
            )
        )

        duplicates = result.get("duplicate_groups") or []
        if duplicates:
            print("\n[同名同版本但 document_id 不同的文件]")
            print(json.dumps(duplicates, ensure_ascii=False, indent=2))

        print("\n[Chunk 读取策略]")
        print(
            json.dumps(
                {
                    "chunk_mode": result.get("chunk_mode"),
                    "detail_queries": result.get("detail_queries") or [],
                    "chunk_count": result.get("retrieved_count", 0),
                },
                ensure_ascii=False,
                indent=2,
            )
        )

        print("\n[送给回答模型的 Chunk]")
        for index, hit in enumerate(result.get("hits") or [], start=1):
            metadata = hit.get("metadata") or {}
            content = str(hit.get("content") or "").replace("\n", " ")
            print(
                f"{index:03d}. file={metadata.get('file_name')} "
                f"version={metadata.get('version')} "
                f"document_id={str(metadata.get('document_id') or '')[:12]} "
                f"location={metadata.get('location')}\n"
                f"     {content[:320]}"
            )
        print()

    print(result["answer"])


if __name__ == "__main__":
    main()
