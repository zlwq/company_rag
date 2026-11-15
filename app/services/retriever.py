import os
import re
from datetime import datetime, timezone
from difflib import SequenceMatcher
from typing import Any

from qdrant_client import models

from app.services.embedder import Embedder
from app.services.qdrant_store import QdrantStore


class Retriever:
    DOCUMENT_FIELDS = {
        "document_id",
        "file_name",
 "file_type",
        "source_path",
        "file_size",
        "file_hash",
        "source_modified_at",
        "source_created_at",
        "project_name",
        "document_type",
        "version",
        "project_stage",
        "customer_name",
        "document_date",
        "keywords",
        "summary",
        "access_scope",
        "is_active",
    }

    HARD_FILTER_FIELDS = {
        "project_name",
        "document_type",
        "version",
        "customer_name",
        "file_name",
        "file_type",
        "project_stage",
        "document_date",
        "source_created_at",
        "source_modified_at",
        "access_scope",
        "is_active",
    }

    SOFT_FILTER_FIELDS = {
        "project_name",
        "document_type",
        "version",
        "customer_name",
        "file_name",
        "file_type",
        "project_stage",
        "document_date",
        "keywords",
        "summary",
    }

    SORT_FIELDS = {
        "document_date",
        "source_created_at",
        "source_modified_at",
    }

    def __init__(self) -> None:
        self._embedder: Embedder | None = None
        self.store = QdrantStore()
        self.soft_match_min = float(os.getenv("RETRIEVAL_SOFT_MATCH_MIN", "0.45"))
        self.chunk_candidates_per_query = int(
            os.getenv("RETRIEVAL_CHUNK_CANDIDATES_PER_QUERY", "16")
        )
        self.chunks_per_document = int(
            os.getenv("RETRIEVAL_CHUNKS_PER_DOCUMENT", "6")
        )
        score_threshold = os.getenv("RETRIEVAL_SCORE_THRESHOLD", "").strip()
        self.score_threshold = float(score_threshold) if score_threshold else None

    @property
    def embedder(self) -> Embedder:
        # metadata_only / none 路由不需要向量模型；只有 relevant chunk 检索时才加载。
        if self._embedder is None:
            self._embedder = Embedder()
        return self._embedder

    @staticmethod
    def _as_list(value: Any) -> list[Any]:
        if value is None:
            return []
        if isinstance(value, (list, tuple, set)):
            return [item for item in value if item not in (None, "")]
        return [value]

    @staticmethod
    def _normalize_text(value: Any) -> str:
        text = str(value or "").lower()
        return re.sub(r"[\s\-_—–·•,，。；;：:、/\\()（）\[\]【】]+", "", text)

    @classmethod
    def _char_ngrams(cls, text: Any, n: int = 2) -> set[str]:
        normalized = cls._normalize_text(text)
        if not normalized:
            return set()
        if len(normalized) <= n:
            return {normalized}
        return {normalized[i : i + n] for i in range(len(normalized) - n + 1)}

    @classmethod
    def _fuzzy_score(cls, query: Any, target: Any) -> float:
        q = cls._normalize_text(query)
        t = cls._normalize_text(target)
        if not q or not t:
            return 0.0
        if q == t or q in t:
            return 1.0
        if t in q and len(t) >= 3:
            return 0.94

        qg = cls._char_ngrams(q)
        tg = cls._char_ngrams(t)
        overlap = len(qg & tg) / len(qg) if qg else 0.0
        sequence = SequenceMatcher(None, q, t).ratio()
        return max(overlap, sequence * 0.9)

    @staticmethod
    def _exact_equal(left: Any, right: Any) -> bool:
        if isinstance(left, bool) or isinstance(right, bool):
            return left is right
        return str(left).strip().casefold() == str(right).strip().casefold()

    def _all_documents(self) -> list[dict[str, Any]]:
        if not self.store.collection_exists():
            return []

        payload_fields = sorted(self.DOCUMENT_FIELDS)
        by_document: dict[str, dict[str, Any]] = {}
        offset = None

        while True:
            points, next_offset = self.store.client.scroll(
                collection_name=self.store.collection_name,
                limit=256,
                offset=offset,
                with_payload=payload_fields,
                with_vectors=False,
            )

            for point in points:
                metadata = dict(point.payload or {})
                document_id = metadata.get("document_id")
                if not document_id:
                    continue
                document_id = str(document_id)
                if document_id not in by_document:
                    by_document[document_id] = {
                        "document_id": document_id,
                        "metadata": metadata,
                    }

            if next_offset is None:
                break
            offset = next_offset

        return list(by_document.values())

    @classmethod
    def _criterion_score(
        cls,
        metadata: dict[str, Any],
        fields: list[str],
        terms: list[str],
    ) -> float:
        best = 0.0
        for field in fields:
            if field not in cls.SOFT_FILTER_FIELDS:
                continue
            raw = metadata.get(field)
            targets = raw if isinstance(raw, list) else [raw]
            for term in terms:
                for target in targets:
                    if target in (None, ""):
                        continue
                    best = max(best, cls._fuzzy_score(term, target))
        return best

    def soft_filter_documents(
        self,
        criteria: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        """模糊过滤只做一件事：按 LLM 给出的属性条件筛选文档，不猜、不改条件。"""
        documents = self._all_documents()
        if not criteria:
            return documents

        selected = []
        for document in documents:
            metadata = document["metadata"]
            criterion_scores = []
            passed = True

            for criterion in criteria:
                fields = [
                    str(field)
                    for field in self._as_list(criterion.get("fields"))
                    if str(field) in self.SOFT_FILTER_FIELDS
                ]
                terms = [
                    str(term).strip()
                    for term in self._as_list(criterion.get("terms"))
                    if str(term).strip()
                ]
                if not fields or not terms:
                    continue

                score = self._criterion_score(metadata, fields, terms)
                criterion_scores.append(
                    {"fields": fields, "terms": terms, "score": round(score, 4)}
                )
                if score < self.soft_match_min:
                    passed = False
                    break

            if not passed:
                continue

            average = (
                sum(item["score"] for item in criterion_scores) / len(criterion_scores)
                if criterion_scores
                else 1.0
            )
            selected.append(
                {
                    **document,
                    "soft_score": average,
                    "soft_matches": criterion_scores,
                }
            )

        selected.sort(key=lambda item: item.get("soft_score", 0.0), reverse=True)
        return selected

    def hard_filter_documents(
        self,
        documents: list[dict[str, Any]],
        filters: dict[str, Any] | None,
        required_fields: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        """只做 metadata 机械筛选：精确等值 + 指定字段必须存在。"""
        cleaned = {
            key: self._as_list(value)
            for key, value in (filters or {}).items()
            if key in self.HARD_FILTER_FIELDS and self._as_list(value)
        }
        required = [
            str(field)
            for field in (required_fields or [])
            if str(field) in self.HARD_FILTER_FIELDS
        ]

        selected = []
        for document in documents:
            metadata = document.get("metadata") or {}

            if any(metadata.get(field) in (None, "", []) for field in required):
                continue

            ok = True
            for field, expected_values in cleaned.items():
                actual_values = self._as_list(metadata.get(field))
                if not actual_values:
                    ok = False
                    break
                if not any(
                    self._exact_equal(actual, expected)
                    for actual in actual_values
                    for expected in expected_values
                ):
                    ok = False
                    break
            if ok:
                selected.append(document)
        return selected

    @staticmethod
    def _parse_sort_value(value: Any) -> float | None:
        if value in (None, ""):
            return None
        text = str(value).strip()
        try:
            parsed = datetime.fromisoformat(text.replace("Z", "+00:00"))
        except ValueError:
            return None
        if parsed.tzinfo is None:
            parsed = parsed.replace(tzinfo=timezone.utc)
        return parsed.timestamp()

    @staticmethod
    def _document_signature(document: dict[str, Any]) -> tuple[str, str]:
        metadata = document.get("metadata") or {}
        return (
            str(metadata.get("file_name") or ""),
            str(metadata.get("version") or ""),
        )

    def sort_documents(
        self,
        documents: list[dict[str, Any]],
        field: str,
        order: str = "asc",
        limit: int | None = None,
        distinct_by: str = "none",
    ) -> tuple[
        list[dict[str, Any]],
        list[dict[str, Any]],
        list[dict[str, Any]],
    ]:
        if field not in self.SORT_FIELDS:
            raise ValueError(f"不支持的排序字段: {field}")
        if distinct_by not in {"none", "version", "file_name"}:
            raise ValueError(f"不支持的 distinct_by: {distinct_by}")

        valid = []
        missing_sort_value = []
        missing_distinct_value = []
        for document in documents:
            metadata = document.get("metadata") or {}
            value = self._parse_sort_value(metadata.get(field))
            if value is None:
                missing_sort_value.append(document)
                continue
            if distinct_by != "none" and metadata.get(distinct_by) in (None, ""):
                missing_distinct_value.append(document)
                continue
            valid.append((value, document))

        reverse = str(order).lower() == "desc"
        valid.sort(key=lambda item: item[0], reverse=reverse)
        ordered = [document for _, document in valid]

        if limit is None or limit <= 0:
            return ordered, missing_sort_value, missing_distinct_value

        if distinct_by != "none":
            selected_values = []
            for document in ordered:
                value = str((document.get("metadata") or {}).get(distinct_by))
                marker = value.casefold()
                if marker not in selected_values:
                    selected_values.append(marker)
                if len(selected_values) >= limit:
                    break
            selected = [
                document
                for document in ordered
                if str((document.get("metadata") or {}).get(distinct_by)).casefold()
                in selected_values
            ]
            return selected, missing_sort_value, missing_distinct_value

        if len(ordered) <= limit:
            return ordered, missing_sort_value, missing_distinct_value

        selected = ordered[:limit]
        signatures = {self._document_signature(document) for document in selected}

        # 选中的文件若存在同名同版本但 document_id 不同的副本，全部保留。
        for document in ordered[limit:]:
            if self._document_signature(document) in signatures:
                selected.append(document)

        return selected, missing_sort_value, missing_distinct_value

    @classmethod
    def duplicate_groups(
        cls,
        documents: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        groups: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for document in documents:
            signature = cls._document_signature(document)
            groups.setdefault(signature, []).append(document)

        result = []
        for (file_name, version), items in groups.items():
            if len(items) <= 1:
                continue
            hashes = [
                str((item.get("metadata") or {}).get("file_hash") or "").strip()
                for item in items
            ]
            known_hashes = {value for value in hashes if value}
            if hashes and all(hashes) and len(known_hashes) == 1:
                content_relation = "identical"
            elif len(known_hashes) >= 2:
                content_relation = "different"
            else:
                content_relation = "unknown"

            result.append(
                {
                    "file_name": file_name,
                    "version": version or None,
                    "content_relation": content_relation,
                    "documents": [
                        {
                            "document_id": item["document_id"],
                            "source_path": (item.get("metadata") or {}).get("source_path"),
                            "file_hash": (item.get("metadata") or {}).get("file_hash"),
                        }
                        for item in items
                    ],
                }
            )
        return result

    @staticmethod
    def _part_number(chunk_key: str) -> int:
        match = re.search(r":part:(\d+)$", str(chunk_key or ""))
        return int(match.group(1)) if match else 1

    @classmethod
    def _chunk_sort_key(cls, payload: dict[str, Any]) -> tuple:
        location = payload.get("location") or {}
        part = cls._part_number(payload.get("chunk_key"))
        if location.get("paragraph") is not None:
            return (0, int(location["paragraph"]), part)
        if location.get("table") is not None and location.get("row") is not None:
            return (1, int(location["table"]), int(location["row"]), part)
        if location.get("page") is not None:
            return (2, int(location["page"]), part)
        if location.get("slide") is not None:
            return (3, int(location["slide"]), part)
        if location.get("sheet") is not None and location.get("row") is not None:
            return (4, str(location["sheet"]), int(location["row"]), part)
        if location.get("line_start") is not None:
            return (5, int(location["line_start"]), part)
        return (9, str(payload.get("chunk_key") or ""))

    def _query_points(
        self,
        vector: list[float],
        limit: int,
        query_filter: models.Filter,
    ) -> list[Any]:
        kwargs = {
            "collection_name": self.store.collection_name,
            "limit": limit,
            "with_payload": True,
            "with_vectors": False,
            "query_filter": query_filter,
        }
        if self.score_threshold is not None:
            kwargs["score_threshold"] = self.score_threshold

        if hasattr(self.store.client, "query_points"):
            response = self.store.client.query_points(query=vector, **kwargs)
            return list(response.points)
        return list(self.store.client.search(query_vector=vector, **kwargs))

    @staticmethod
    def _document_filter(document_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

    def _all_chunks_for_document(self, document_id: str) -> list[dict[str, Any]]:
        points = self.store.get_document_points(document_id)
        ordered = sorted(points.items(), key=lambda item: self._chunk_sort_key(item[1]))
        hits = []
        for point_id, payload in ordered:
            content = str(payload.get("content") or "").strip()
            if not content:
                continue
            hits.append(
                {
                    "point_id": str(point_id),
                    "score": 1.0,
                    "content": content,
                    "metadata": dict(payload),
                    "matched_queries": ["完整文档"],
                }
            )
        return hits

    @classmethod
    def _text_relevance(cls, query: str, text: str) -> float:
        q = cls._char_ngrams(query)
        t = cls._char_ngrams(text)
        if not q or not t:
            return 0.0
        return len(q & t) / len(q)

    def _relevant_chunks_for_document(
        self,
        document_id: str,
        detail_queries: list[str],
    ) -> list[dict[str, Any]]:
        queries = [query.strip() for query in detail_queries if query.strip()]
        if not queries:
            return self._all_chunks_for_document(document_id)

        vectors = self.embedder.embed(queries)
        merged: dict[str, dict[str, Any]] = {}
        per_query_ranked: list[list[dict[str, Any]]] = []
        document_filter = self._document_filter(document_id)

        # 每条 detail_query 独立向量召回，保持原有 vector + lexical 混合评分。
        for query, vector in zip(queries, vectors):
            points = self._query_points(
                vector=vector,
                limit=self.chunk_candidates_per_query,
                query_filter=document_filter,
            )
            query_hits: list[dict[str, Any]] = []

            for point in points:
                payload = dict(point.payload or {})
                content = str(payload.get("content") or "").strip()
                if not content:
                    continue

                point_id = str(point.id)
                vector_score = float(point.score)
                lexical_score = self._text_relevance(query, content)
                score = vector_score * 0.78 + lexical_score * 0.22

                query_hit = {
                    "point_id": point_id,
                    "score": score,
                    "content": content,
                    "metadata": payload,
                    "matched_queries": [query],
                }
                query_hits.append(query_hit)

                current = merged.get(point_id)
                if current is None:
                    merged[point_id] = dict(query_hit)
                else:
                    current["score"] = max(current["score"], score)
                    if query not in current["matched_queries"]:
                        current["matched_queries"].append(query)

            query_hits.sort(
                key=lambda item: item["score"],
                reverse=True,
            )
            per_query_ranked.append(query_hits)

        # 第一阶段：优先保证每个 detail_query 都有代表证据。
        # 如果同一个 chunk 同时命中多条 query，它可以一次覆盖这些 query，
        # 不会为了凑数量强行加入低质量的重复证据。
        selected: list[dict[str, Any]] = []
        selected_ids: set[str] = set()

        for query, query_hits in zip(queries, per_query_ranked):
            already_covered = any(
                query in merged[point_id].get("matched_queries", [])
                for point_id in selected_ids
            )
            if already_covered:
                continue

            for hit in query_hits:
                point_id = hit["point_id"]
                if point_id in selected_ids:
                    continue
                selected.append(merged[point_id])
                selected_ids.add(point_id)
                break

        # 第二阶段：剩余位置仍按原来的混合分数全局补齐。
        # 多查询时允许结果数至少等于 query 数，避免 7~8 个独立问题被固定 6 个 chunk 截断。
        limit = max(self.chunks_per_document, len(queries))
        ranked_all = sorted(
            merged.values(),
            key=lambda item: item["score"],
            reverse=True,
        )

        for hit in ranked_all:
            if len(selected) >= limit:
                break
            point_id = hit["point_id"]
            if point_id in selected_ids:
                continue
            selected.append(hit)
            selected_ids.add(point_id)

        return selected

    def load_chunks(
        self,
        documents: list[dict[str, Any]],
        mode: str,
        detail_queries: list[str] | None = None,
    ) -> list[dict[str, Any]]:
        if mode == "metadata_only":
            return []

        hits = []
        for document in documents:
            document_id = document["document_id"]
            if mode == "all":
                document_hits = self._all_chunks_for_document(document_id)
            else:
                document_hits = self._relevant_chunks_for_document(
                    document_id,
                    detail_queries or [],
                )

            for hit in document_hits:
                hit["selected_document"] = {
                    "document_id": document_id,
                    "metadata": document.get("metadata") or {},
                }
                hits.append(hit)
        return hits

    def new_session(
        self,
        explicit_filters: dict[str, Any] | None = None,
    ) -> "DocumentSearchSession":
        return DocumentSearchSession(self, explicit_filters=explicit_filters)


class DocumentSearchSession:
    def __init__(
        self,
        retriever: Retriever,
        explicit_filters: dict[str, Any] | None = None,
    ) -> None:
        self.retriever = retriever
        self.explicit_filters = {
            key: value
            for key, value in (explicit_filters or {}).items()
            if key in retriever.HARD_FILTER_FIELDS and value not in (None, "", [])
        }
        self.documents: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self.finalized = False
        self.chunk_mode = "metadata_only"
        self.detail_queries: list[str] = []

    @staticmethod
    def _summary(documents: list[dict[str, Any]]) -> list[dict[str, Any]]:
        result = []
        for document in documents:
            metadata = document.get("metadata") or {}
            result.append(
                {
                    "document_id": document.get("document_id"),
                    "file_name": metadata.get("file_name"),
                    "project_name": metadata.get("project_name"),
                    "document_type": metadata.get("document_type"),
                    "version": metadata.get("version"),
                    "document_date": metadata.get("document_date"),
                    "source_created_at": metadata.get("source_created_at"),
                    "source_modified_at": metadata.get("source_modified_at"),
                    "source_path": metadata.get("source_path"),
                    "file_hash": metadata.get("file_hash"),
                }
            )
        return result

    def execute(self, name: str, arguments: dict[str, Any]) -> dict[str, Any]:
        if name == "soft_search_documents":
            criteria = arguments.get("criteria") or []
            self.documents = self.retriever.soft_filter_documents(criteria)
            if self.explicit_filters:
                self.documents = self.retriever.hard_filter_documents(
                    self.documents,
                    self.explicit_filters,
                )
            result = {
                "count": len(self.documents),
                "documents": self._summary(self.documents),
            }

        elif name == "hard_filter_documents":
            filters = arguments.get("filters") or {}
            required_fields = [
                str(field)
                for field in (arguments.get("required_fields") or [])
                if str(field) in self.retriever.HARD_FILTER_FIELDS
            ]
            self.documents = self.retriever.hard_filter_documents(
                self.documents,
                filters,
                required_fields=required_fields,
            )
            result = {
                "count": len(self.documents),
                "documents": self._summary(self.documents),
                "required_fields": required_fields,
            }

        elif name == "sort_documents":
            field = str(arguments.get("field") or "")
            order = str(arguments.get("order") or "asc")
            limit = arguments.get("limit")
            distinct_by = str(arguments.get("distinct_by") or "none")
            self.documents, missing, missing_distinct = self.retriever.sort_documents(
                self.documents,
                field=field,
                order=order,
                limit=int(limit) if limit is not None else None,
                distinct_by=distinct_by,
            )
            result = {
                "count": len(self.documents),
                "documents": self._summary(self.documents),
                "missing_sort_value": self._summary(missing),
                "missing_distinct_value": self._summary(missing_distinct),
            }

        elif name == "finalize_documents":
            self.finalized = True
            requested_mode = str(arguments.get("chunk_mode") or "metadata_only")
            self.chunk_mode = (
                requested_mode
                if requested_mode in {"metadata_only", "all", "relevant"}
                else "metadata_only"
            )
            self.detail_queries = [
                str(item).strip()
                for item in (arguments.get("detail_queries") or [])
                if str(item).strip()
            ][:8]
            result = {
                "count": len(self.documents),
                "documents": self._summary(self.documents),
                "chunk_mode": self.chunk_mode,
                "detail_queries": self.detail_queries,
                "duplicate_groups": self.retriever.duplicate_groups(self.documents),
            }

        else:
            raise ValueError(f"未知检索工具: {name}")

        self.trace.append(
            {
                "tool": name,
                "arguments": arguments,
                "result": result,
            }
        )
        return result
