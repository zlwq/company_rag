import os
from typing import Iterable

import numpy as np
from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, model_name: str | None = None, device: str | None = None):
        self.model_name = model_name or os.getenv("EMBEDDING_MODEL", "BAAI/bge-m3")
        self.device = device or os.getenv("EMBEDDING_DEVICE") or None
        # 只从本机缓存或本地目录加载，不在查询时访问 Hugging Face。
        self.model = SentenceTransformer(
            self.model_name,
            device=self.device,
            local_files_only=True,
        )
        # 长表格直接编码会触发注意力层显存暴涨。
        self.model.max_seq_length = min(self.model.max_seq_length or 512, 512)

    @property
    def dimension(self) -> int:
        if hasattr(self.model, "get_embedding_dimension"):
            return int(self.model.get_embedding_dimension())
        return int(self.model.get_sentence_embedding_dimension())

    def embed(self, texts: Iterable[str], batch_size: int = 1) -> list[list[float]]:
        texts = list(texts)
        if not texts:
            return []

        # 只在生成向量时分段；Qdrant 中仍保存原文完整的单个 chunk。
        window_size = min(self.model.max_seq_length - 2, 384)
        tokenizer = self.model.tokenizer
        windows = []
        owners = []
        for index, value in enumerate(texts):
            token_ids = tokenizer.encode(value, add_special_tokens=False)
            if len(token_ids) <= window_size:
                windows.append(value)
                owners.append(index)
            else:
                for start in range(0, len(token_ids), window_size):
                    windows.append(tokenizer.decode(token_ids[start:start + window_size], skip_special_tokens=True))
                    owners.append(index)

        vectors = self.model.encode(
            windows,
            batch_size=batch_size,
            normalize_embeddings=True,
            show_progress_bar=len(windows) > batch_size,
            convert_to_numpy=True,
        )
        totals = np.zeros((len(texts), vectors.shape[1]), dtype=np.float32)
        for owner, vector in zip(owners, vectors):
            totals[owner] += vector
        norms = np.linalg.norm(totals, axis=1, keepdims=True)
        totals /= np.maximum(norms, 1e-12)
        return totals.tolist()

