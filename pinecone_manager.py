"""Модуль для чтения и записи данных в Pinecone."""

from __future__ import annotations

import math
import os
import uuid
from dataclasses import dataclass
from typing import Any, Literal, Mapping, Sequence

from dotenv import load_dotenv
from pinecone import AsyncPinecone, SparseValues, Vector

load_dotenv()

# Порог косинусного сходства для долговременной памяти чат-бота.
# score >= порога — дубликат или близкая вариация (пропуск или обновление слота).
# score < порога — новая информация (создание новой записи).
COSINE_SIMILARITY_THRESHOLD: float = 0.85

DuplicatePolicy = Literal["skip", "update"]
MemoryAction = Literal["inserted", "skipped", "updated"]


@dataclass(slots=True)
class SearchHit:
    """Результат поиска по индексу."""

    id: str
    score: float
    metadata: dict[str, Any] | None = None
    values: list[float] | None = None
    fields: dict[str, Any] | None = None


@dataclass(slots=True)
class UpsertResult:
    """Результат операции записи."""

    count: int


@dataclass(slots=True)
class MemoryWriteResult:
    """Результат записи в долговременную память с проверкой дубликатов."""

    action: MemoryAction
    memory_id: str
    similarity: float | None = None
    matched_id: str | None = None
    upsert_result: UpsertResult | None = None


class PineconeManager:
    """
    Менеджер операций чтения/записи для Pinecone.

    Поддерживает два типа индексов:
    - integrated — текстовые документы через upsert_records / search по тексту
    - standard — готовые векторы через upsert / query

    Пример:
        async with PineconeManager(index_name="my-index") as pc:
            await pc.upsert_document("doc-1", "Текст документа", category="faq")
            hits = await pc.search_by_text("как это работает?", top_k=5)
    """

    def __init__(
        self,
        index_name: str | None = None,
        *,
        api_key: str | None = None,
        namespace: str = "__default__",
        text_field: str = "chunk_text",
        integrated: bool | None = None,
    ) -> None:
        resolved_index = (index_name or os.getenv("PINECONE_INDEX_NAME", "")).strip()
        if not resolved_index:
            raise ValueError(
                "index_name не задан. Передайте имя или добавьте PINECONE_INDEX_NAME в .env"
            )

        if integrated is None:
            env_integrated = os.getenv("PINECONE_INTEGRATED", "false").strip().lower()
            integrated = env_integrated in {"1", "true", "yes", "on"}

        self.index_name = resolved_index
        self.namespace = namespace.strip() or "__default__"
        self.text_field = text_field
        self.integrated = integrated

        self._api_key = api_key or os.getenv("PINECONE_API_KEY")
        if not self._api_key:
            raise ValueError(
                "PINECONE_API_KEY не задан. Передайте api_key или добавьте ключ в .env"
            )

        self._client: AsyncPinecone | None = None
        self._index = None

    async def __aenter__(self) -> PineconeManager:
        await self.connect()
        return self

    async def __aexit__(self, exc_type, exc, tb) -> None:
        await self.close()

    async def connect(self) -> None:
        """Подключиться к Pinecone и открыть индекс."""
        if self._client is not None:
            return

        self._client = AsyncPinecone(api_key=self._api_key)
        self._index = await self._client.index(name=self.index_name)

    async def close(self) -> None:
        """Закрыть клиент Pinecone."""
        if self._client is not None:
            await self._client.close()
            self._client = None
            self._index = None

    @property
    def index(self):
        if self._index is None:
            raise RuntimeError("Клиент не подключён. Вызовите connect() или используйте async with.")
        return self._index

    # ------------------------------------------------------------------
    # Запись
    # ------------------------------------------------------------------

    async def upsert_vector(
        self,
        vector_id: str,
        vector: Sequence[float],
        metadata: Mapping[str, Any] | None = None,
        *,
        check_similarity: bool = True,
        sparse_values: Mapping[str, Sequence[float | int]] | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """
        Запись вектора в Pinecone с проверкой косинусного сходства.

        Args:
            vector_id: Уникальный идентификатор вектора.
            vector: Вектор для записи.
            metadata: Метаданные (опционально).
            check_similarity: Проверять ли сходство перед записью.
            sparse_values: Sparse-компонент для hybrid-индекса.
            metadata_filter: Фильтр по metadata при поиске похожих векторов.

        Returns:
            Словарь с результатом:
            - action: inserted | updated | skipped
            - similarity_score: значение сходства (если найдено)
            - existing_id: ID существующего вектора (если найден)
        """
        if self.integrated:
            raise ValueError("upsert_vector доступен только для standard-индексов")

        result: dict[str, Any] = {
            "action": "inserted",
            "similarity_score": None,
            "existing_id": None,
        }

        if check_similarity:
            similar = await self._check_similarity(
                vector,
                exclude_id=vector_id,
                metadata_filter=metadata_filter,
                sparse_vector=sparse_values,
            )
            if similar:
                existing_id = similar["id"]
                result["action"] = "updated"
                result["similarity_score"] = similar["score"]
                result["existing_id"] = existing_id

                await self._upsert_vector_raw(
                    existing_id,
                    vector,
                    metadata=metadata,
                    sparse_values=sparse_values,
                )
                return result

        await self._upsert_vector_raw(
            vector_id,
            vector,
            metadata=metadata,
            sparse_values=sparse_values,
        )
        return result

    async def _upsert_vector_raw(
        self,
        vector_id: str,
        vector: Sequence[float],
        *,
        metadata: Mapping[str, Any] | None = None,
        sparse_values: Mapping[str, Sequence[float | int]] | None = None,
    ) -> UpsertResult:
        """Записать вектор без проверки сходства."""
        payload: dict[str, Any] = {
            "id": vector_id,
            "values": list(vector),
        }
        if metadata:
            payload["metadata"] = dict(metadata)
        if sparse_values:
            payload["sparse_values"] = {
                "indices": list(sparse_values["indices"]),
                "values": list(sparse_values["values"]),
            }

        response = await self.index.upsert(
            vectors=[payload],
            namespace=self.namespace,
        )
        return UpsertResult(count=response.upserted_count or 1)

    async def upsert_vectors(
        self,
        vectors: Sequence[Vector | Mapping[str, Any] | tuple],
        *,
        batch_size: int | None = None,
    ) -> UpsertResult:
        """
        Пакетная запись векторов в standard-индекс.

        Каждый элемент может быть:
        - Vector
        - dict с ключами id, values, metadata, sparse_values
        - tuple (id, values) или (id, values, metadata)
        """
        kwargs: dict[str, Any] = {
            "vectors": vectors,
            "namespace": self.namespace,
        }
        if batch_size is not None:
            kwargs["batch_size"] = batch_size

        response = await self.index.upsert(**kwargs)
        return UpsertResult(count=response.upserted_count or len(vectors))

    async def upsert_document(
        self,
        document_id: str,
        text: str,
        **fields: Any,
    ) -> UpsertResult:
        """Записать один текстовый документ в integrated-индекс."""
        record = self._build_record(document_id, text, fields)
        return await self.upsert_documents([record])

    async def upsert_documents(
        self,
        records: Sequence[Mapping[str, Any]],
    ) -> UpsertResult:
        """
        Пакетная запись документов в integrated-индекс.

        Каждая запись должна содержать _id/id и текстовое поле (по умолчанию chunk_text).
        Дополнительные поля сохраняются как metadata для фильтрации.
        """
        if not records:
            raise ValueError("records не может быть пустым")

        normalized = [self._normalize_record(record) for record in records]
        response = await self.index.upsert_records(
            namespace=self.namespace,
            records=normalized,
        )
        count = getattr(response, "record_count", None) or getattr(response, "upserted_count", None)
        return UpsertResult(count=count or len(normalized))

    async def remember_text(
        self,
        text: str,
        *,
        memory_id: str | None = None,
        on_duplicate: DuplicatePolicy = "update",
        similarity_threshold: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        **fields: Any,
    ) -> MemoryWriteResult:
        """
        Записать фрагмент в долговременную память (integrated-индекс).

        Перед записью ищет ближайший сохранённый фрагмент и сравнивает
        косинусное сходство с порогом COSINE_SIMILARITY_THRESHOLD.
        """
        if not self.integrated:
            raise ValueError("remember_text доступен только для integrated-индексов")

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else COSINE_SIMILARITY_THRESHOLD
        )
        hits = await self.search_by_text(
            text,
            top_k=1,
            metadata_filter=metadata_filter,
        )
        best_hit = hits[0] if hits else None
        similarity = best_hit.score if best_hit else None

        if best_hit is not None and similarity is not None and similarity >= threshold:
            return await self._handle_duplicate_memory(
                duplicate_id=best_hit.id,
                similarity=similarity,
                on_duplicate=on_duplicate,
                write=lambda record_id: self.upsert_document(record_id, text, **fields),
                fallback_memory_id=memory_id,
            )

        record_id = memory_id or str(uuid.uuid4())
        upsert_result = await self.upsert_document(record_id, text, **fields)
        return MemoryWriteResult(
            action="inserted",
            memory_id=record_id,
            similarity=similarity,
            upsert_result=upsert_result,
        )

    async def remember_vector(
        self,
        values: Sequence[float],
        *,
        memory_id: str | None = None,
        metadata: Mapping[str, Any] | None = None,
        on_duplicate: DuplicatePolicy = "update",
        similarity_threshold: float | None = None,
        metadata_filter: Mapping[str, Any] | None = None,
        sparse_values: Mapping[str, Sequence[float | int]] | None = None,
    ) -> MemoryWriteResult:
        """
        Записать вектор в долговременную память (standard-индекс).

        Перед записью ищет ближайший сохранённый вектор и сравнивает
        косинусное сходство с порогом COSINE_SIMILARITY_THRESHOLD.
        """
        if self.integrated:
            raise ValueError("remember_vector доступен только для standard-индексов")

        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else COSINE_SIMILARITY_THRESHOLD
        )
        hits = await self.search_by_vector(
            values,
            top_k=1,
            metadata_filter=metadata_filter,
            include_values=False,
            include_metadata=True,
            sparse_vector=sparse_values,
        )
        best_hit = hits[0] if hits else None
        similarity = best_hit.score if best_hit else None

        if best_hit is not None and similarity is not None and similarity >= threshold:
            return await self._handle_duplicate_memory(
                duplicate_id=best_hit.id,
                similarity=similarity,
                on_duplicate=on_duplicate,
                write=lambda record_id: self._upsert_vector_raw(
                    record_id,
                    values,
                    metadata=metadata,
                    sparse_values=sparse_values,
                ),
                fallback_memory_id=memory_id,
            )

        record_id = memory_id or str(uuid.uuid4())
        upsert_result = await self._upsert_vector_raw(
            record_id,
            values,
            metadata=metadata,
            sparse_values=sparse_values,
        )
        return MemoryWriteResult(
            action="inserted",
            memory_id=record_id,
            similarity=similarity,
            upsert_result=upsert_result,
        )

    # ------------------------------------------------------------------
    # Чтение
    # ------------------------------------------------------------------

    async def search_by_text(
        self,
        query: str,
        *,
        top_k: int = 10,
        metadata_filter: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        rerank: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """
        Семантический поиск по тексту в integrated-индексе.

        Pinecone сам эмбеддит query на стороне сервера.
        """
        if not self.integrated:
            raise ValueError("search_by_text доступен только для integrated-индексов")

        response = await self.index.search(
            namespace=self.namespace,
            top_k=top_k,
            inputs={"text": query},
            filter=dict(metadata_filter) if metadata_filter else None,
            fields=list(fields) if fields else None,
            rerank=dict(rerank) if rerank else None,
        )
        return self._parse_search_hits(response)

    async def _embed_text(self, text: str) -> list[float]:
        from openai import AsyncOpenAI

        api_key = os.getenv("OPENAI_API_KEY", "").strip()
        if not api_key:
            raise ValueError("OPENAI_API_KEY не задан в .env")

        client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("OPENAI_BASE_URL") or None,
        )
        model = os.getenv("EMBEDDING_MODEL", "text-embedding-3-small")
        response = await client.embeddings.create(model=model, input=text)
        return response.data[0].embedding

    async def query_by_text(
        self,
        text: str,
        *,
        top_k: int = 10,
        metadata_filter: Mapping[str, Any] | None = None,
        fields: Sequence[str] | None = None,
        rerank: Mapping[str, Any] | None = None,
        vector: Sequence[float] | None = None,
    ) -> list[SearchHit]:
        """Поиск по тексту: integrated — через Pinecone, standard — через vector."""
        if self.integrated:
            return await self.search_by_text(
                text,
                top_k=top_k,
                metadata_filter=metadata_filter,
                fields=fields,
                rerank=rerank,
            )

        if vector is None:
            vector = await self._embed_text(text)

        return await self.search_by_vector(
            vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
        )

    async def search_by_vector(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        metadata_filter: Mapping[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
        sparse_vector: Mapping[str, Sequence[float | int]] | None = None,
    ) -> list[SearchHit]:
        """
        Поиск по готовому вектору.

        - standard-индекс: query()
        - integrated-индекс: search(vector=...)
        """
        filter_payload = dict(metadata_filter) if metadata_filter else None

        if self.integrated:
            response = await self.index.search(
                namespace=self.namespace,
                top_k=top_k,
                vector=list(vector),
                filter=filter_payload,
            )
            return self._parse_search_hits(response)

        query_kwargs: dict[str, Any] = {
            "vector": list(vector),
            "top_k": top_k,
            "namespace": self.namespace,
            "include_values": include_values,
            "include_metadata": include_metadata,
            "filter": filter_payload,
        }
        if sparse_vector:
            query_kwargs["sparse_vector"] = SparseValues(
                indices=list(sparse_vector["indices"]),
                values=list(sparse_vector["values"]),
            )

        response = await self.index.query(**query_kwargs)
        return self._parse_query_matches(response)

    async def query_by_vector(
        self,
        vector: Sequence[float],
        *,
        top_k: int = 10,
        metadata_filter: Mapping[str, Any] | None = None,
        include_values: bool = False,
        include_metadata: bool = True,
        sparse_vector: Mapping[str, Sequence[float | int]] | None = None,
    ) -> list[SearchHit]:
        """Поиск по вектору (алиас search_by_vector)."""
        return await self.search_by_vector(
            vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_values=include_values,
            include_metadata=include_metadata,
            sparse_vector=sparse_vector,
        )

    async def search_by_id(
        self,
        record_id: str,
        *,
        top_k: int = 10,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> list[SearchHit]:
        """Поиск похожих записей относительно существующей записи (integrated-индекс)."""
        if not self.integrated:
            raise ValueError("search_by_id доступен только для integrated-индексов")

        response = await self.index.search(
            namespace=self.namespace,
            top_k=top_k,
            id=record_id,
            filter=dict(metadata_filter) if metadata_filter else None,
        )
        return self._parse_search_hits(response)

    async def fetch(
        self,
        ids: Sequence[str],
    ) -> dict[str, Any]:
        """Получить векторы/записи по ID."""
        response = await self.index.fetch(ids=list(ids), namespace=self.namespace)
        return {
            vector_id: {
                "values": vector.values,
                "metadata": vector.metadata,
                "sparse_values": getattr(vector, "sparse_values", None),
            }
            for vector_id, vector in response.vectors.items()
        }

    async def get_stats(
        self,
        metadata_filter: Mapping[str, Any] | None = None,
    ) -> dict[str, Any]:
        """Статистика индекса: количество векторов, namespaces, dimension."""
        response = await self.index.describe_index_stats(
            filter=dict(metadata_filter) if metadata_filter else None,
        )
        return {
            "dimension": response.dimension,
            "total_vector_count": response.total_vector_count,
            "namespaces": {
                name: {"vector_count": summary.vector_count}
                for name, summary in (response.namespaces or {}).items()
            },
        }

    # ------------------------------------------------------------------
    # Удаление
    # ------------------------------------------------------------------

    async def delete_by_ids(self, ids: Sequence[str]) -> None:
        """Удалить записи по ID."""
        await self.index.delete(ids=list(ids), namespace=self.namespace)

    async def delete_by_filter(self, metadata_filter: Mapping[str, Any]) -> None:
        """Удалить записи по metadata-фильтру."""
        await self.index.delete(
            filter=dict(metadata_filter),
            namespace=self.namespace,
        )

    async def delete_all(self) -> None:
        """Удалить все записи в текущем namespace."""
        await self.index.delete(delete_all=True, namespace=self.namespace)

    # ------------------------------------------------------------------
    # Вспомогательные методы
    # ------------------------------------------------------------------

    @staticmethod
    def cosine_similarity(
        left: Sequence[float],
        right: Sequence[float],
    ) -> float:
        """Посчитать косинусное сходство между двумя dense-векторами."""
        if len(left) != len(right):
            raise ValueError("Векторы должны иметь одинаковую размерность")

        dot = 0.0
        left_norm = 0.0
        right_norm = 0.0
        for left_value, right_value in zip(left, right, strict=True):
            dot += left_value * right_value
            left_norm += left_value * left_value
            right_norm += right_value * right_value

        if left_norm == 0.0 or right_norm == 0.0:
            return 0.0

        return dot / (math.sqrt(left_norm) * math.sqrt(right_norm))

    async def _check_similarity(
        self,
        vector: Sequence[float],
        top_k: int = 5,
        exclude_id: str | None = None,
        *,
        metadata_filter: Mapping[str, Any] | None = None,
        sparse_vector: Mapping[str, Sequence[float | int]] | None = None,
        similarity_threshold: float | None = None,
    ) -> dict[str, Any] | None:
        """
        Проверка косинусного сходства текущего вектора с уже сохранёнными.

        Args:
            vector: Вектор для проверки.
            top_k: Количество наиболее похожих векторов для проверки.
            exclude_id: ID вектора, который нужно исключить из проверки.
            metadata_filter: Фильтр по metadata при поиске.
            sparse_vector: Sparse-компонент для hybrid-индекса.
            similarity_threshold: Порог сходства (по умолчанию COSINE_SIMILARITY_THRESHOLD).

        Returns:
            Словарь с информацией о наиболее похожем векторе, если сходство выше порога,
            иначе None. Содержит ключи: id, score, metadata.
        """
        threshold = (
            similarity_threshold
            if similarity_threshold is not None
            else COSINE_SIMILARITY_THRESHOLD
        )

        # Ищем наиболее похожие векторы
        query_response = await self.query_by_vector(
            vector=vector,
            top_k=top_k,
            metadata_filter=metadata_filter,
            include_metadata=True,
            include_values=False,
            sparse_vector=sparse_vector,
        )

        for hit in query_response:
            if exclude_id and hit.id == exclude_id:
                continue
            if hit.score >= threshold:
                return {
                    "id": hit.id,
                    "score": hit.score,
                    "metadata": dict(hit.metadata or {}),
                }

        return None

    async def _handle_duplicate_memory(
        self,
        *,
        duplicate_id: str,
        similarity: float,
        on_duplicate: DuplicatePolicy,
        write,
        fallback_memory_id: str | None,
    ) -> MemoryWriteResult:
        if on_duplicate == "skip":
            return MemoryWriteResult(
                action="skipped",
                memory_id=fallback_memory_id or duplicate_id,
                similarity=similarity,
                matched_id=duplicate_id,
            )

        upsert_result = await write(duplicate_id)
        return MemoryWriteResult(
            action="updated",
            memory_id=duplicate_id,
            similarity=similarity,
            matched_id=duplicate_id,
            upsert_result=upsert_result,
        )

    def _build_record(
        self,
        document_id: str,
        text: str,
        fields: Mapping[str, Any],
    ) -> dict[str, Any]:
        record: dict[str, Any] = {
            "_id": document_id,
            self.text_field: text,
        }
        record.update(fields)
        return record

    def _normalize_record(self, record: Mapping[str, Any]) -> dict[str, Any]:
        normalized = dict(record)
        record_id = normalized.pop("_id", normalized.pop("id", None))
        if not record_id:
            raise ValueError("Каждая запись должна содержать _id или id")
        normalized["_id"] = str(record_id)
        return normalized

    @staticmethod
    def _parse_search_hits(response: Any) -> list[SearchHit]:
        hits = getattr(getattr(response, "result", None), "hits", None)
        if hits is None:
            hits = getattr(response, "hits", []) or []

        parsed: list[SearchHit] = []
        for hit in hits:
            fields = getattr(hit, "fields", None)
            metadata = getattr(hit, "metadata", None)
            if metadata is None and isinstance(fields, dict):
                metadata = {
                    key: value
                    for key, value in fields.items()
                    if key not in {"id", "_id"}
                }

            parsed.append(
                SearchHit(
                    id=str(getattr(hit, "id", getattr(hit, "_id", ""))),
                    score=float(getattr(hit, "score", 0.0)),
                    metadata=metadata,
                    values=getattr(hit, "values", None),
                    fields=fields if isinstance(fields, dict) else None,
                )
            )
        return parsed

    @staticmethod
    def format_matches(hits: list[SearchHit]) -> dict[str, list[dict[str, Any]]]:
        """Формат ответа Pinecone query: {'matches': [{id, metadata, score, values}, ...]}."""
        return {
            "matches": [
                {
                    "id": hit.id,
                    "metadata": dict(hit.metadata or {}),
                    "score": hit.score,
                    "values": list(hit.values or []),
                }
                for hit in hits
            ]
        }

    @staticmethod
    def _parse_query_matches(response: Any) -> list[SearchHit]:
        return [
            SearchHit(
                id=match.id,
                score=float(match.score or 0.0),
                metadata=match.metadata,
                values=match.values,
            )
            for match in response.matches or []
        ]


if __name__ == "__main__":
    import asyncio
    import pprint

    pinecone_manager = PineconeManager()

    async def _run() -> list[SearchHit]:
        async with pinecone_manager:
            return await pinecone_manager.query_by_text(
                text="Я бы хотел лететь на Марс!"
            )

    result = asyncio.run(_run())
    pprint.pprint(
        PineconeManager.format_matches(result),
        sort_dicts=False,
        width=120,
    )
