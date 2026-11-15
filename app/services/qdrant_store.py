import os
from typing import Any

from qdrant_client import QdrantClient, models


class QdrantStore:
    def __init__(self, collection_name: str | None = None):
        self.collection_name = collection_name or os.getenv(
            "QDRANT_COLLECTION", "company_documents"
        )
        url = os.getenv("QDRANT_URL", "http://localhost:6333")
        api_key = os.getenv("QDRANT_API_KEY") or None
        self.upsert_batch_size = int(os.getenv("QDRANT_UPSERT_BATCH_SIZE", "64"))
        self.payload_batch_size = int(os.getenv("QDRANT_PAYLOAD_BATCH_SIZE", "256"))
        self.client = QdrantClient(url=url, api_key=api_key)

    def collection_exists(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def ensure_collection(self, vector_size: int) -> None:
        if self.collection_exists():
            return

        self.client.create_collection(
            collection_name=self.collection_name,
            vectors_config=models.VectorParams(
                size=vector_size,
                distance=models.Distance.COSINE,
            ),
        )

        for field_name in [
            "document_id",
            "project_name",
            "document_type",
            "version",
            "file_name",
        ]:
            self.client.create_payload_index(
                collection_name=self.collection_name,
                field_name=field_name,
                field_schema=models.PayloadSchemaType.KEYWORD,
            )

    def _document_filter(self, document_id: str) -> models.Filter:
        return models.Filter(
            must=[
                models.FieldCondition(
                    key="document_id",
                    match=models.MatchValue(value=document_id),
                )
            ]
        )

    def get_document_points(self, document_id: str) -> dict[str, dict[str, Any]]:
        if not self.collection_exists():
            return {}

        result: dict[str, dict[str, Any]] = {}
        offset = None

        while True:
            points, next_offset = self.client.scroll(
                collection_name=self.collection_name,
                scroll_filter=self._document_filter(document_id),
                limit=256,
                offset=offset,
                with_payload=True,
                with_vectors=False,
            )

            for point in points:
                result[str(point.id)] = dict(point.payload or {})

            if next_offset is None:
                break
            offset = next_offset

        return result

    def get_document_file_hash(self, document_id: str) -> str | None:
        if not self.collection_exists():
            return None

        points, _ = self.client.scroll(
            collection_name=self.collection_name,
            scroll_filter=self._document_filter(document_id),
            limit=1,
            with_payload=["file_hash"],
            with_vectors=False,
        )

        if not points:
            return None
        return (points[0].payload or {}).get("file_hash")

    def upsert_chunks(
        self,
        chunks: list[dict[str, Any]],
        vectors: list[list[float]],
        batch_size: int | None = None,
    ) -> None:
        if not chunks:
            return
        if len(chunks) != len(vectors):
            raise ValueError("chunks 和 vectors 数量不一致")

        batch_size = batch_size or self.upsert_batch_size

        for start in range(0, len(chunks), batch_size):
            batch_chunks = chunks[start : start + batch_size]
            batch_vectors = vectors[start : start + batch_size]

            points = [
                models.PointStruct(
                    id=chunk["chunk_id"],
                    vector=vector,
                    payload=chunk,
                )
                for chunk, vector in zip(batch_chunks, batch_vectors)
            ]

            self.client.upsert(
                collection_name=self.collection_name,
                points=points,
                wait=True,
            )

    def delete_points(self, point_ids: list[str]) -> None:
        if not point_ids:
            return

        for start in range(0, len(point_ids), self.payload_batch_size):
            batch_ids = point_ids[start : start + self.payload_batch_size]
            self.client.delete(
                collection_name=self.collection_name,
                points_selector=models.PointIdsList(points=batch_ids),
                wait=True,
            )

    def set_payload(self, point_ids: list[str], payload: dict[str, Any]) -> None:
        if not point_ids or not payload:
            return

        for start in range(0, len(point_ids), self.payload_batch_size):
            batch_ids = point_ids[start : start + self.payload_batch_size]
            self.client.set_payload(
                collection_name=self.collection_name,
                payload=payload,
                points=batch_ids,
                wait=True,
            )
