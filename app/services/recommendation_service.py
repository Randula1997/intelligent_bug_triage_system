from __future__ import annotations

from collections.abc import Callable, Iterable

from app.core.config import Settings
from app.db.milvus import MilvusRepository
from app.models.schemas import BugDatasetRecord
from app.models.schemas import ExpertiseRecord
from app.services.classification_service import ClassificationService
from app.services.embedding_service import EmbeddingService


class RecommendationService:
    upload_batch_size = 32

    def __init__(
        self,
        *,
        settings: Settings,
        embedding_service: EmbeddingService,
        classification_service: ClassificationService,
        milvus_repository: MilvusRepository,
    ) -> None:
        self.settings = settings
        self.embedding_service = embedding_service
        self.classification_service = classification_service
        self.milvus_repository = milvus_repository

    def train_classifier(
        self,
        records: list[BugDatasetRecord],
        source_name: str,
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> dict[str, object]:
        return self.classification_service.fine_tune(
            records,
            source_name,
            progress_callback=progress_callback,
        )

    def upload_expertise(
        self,
        records: list[ExpertiseRecord] | list[BugDatasetRecord],
        progress_callback: Callable[[str, float, str], None] | None = None,
    ) -> dict[str, int | str]:
        developer_names: list[str] = []
        original_texts: list[str] = []

        for record in records:
            for history_text in self._extract_record_texts(record):
                developer_names.append(record.developer_name)
                original_texts.append(history_text)

        if not original_texts:
            raise ValueError("No bug history text was found in the uploaded dataset.")

        developer_names, original_texts, skipped_records = self._filter_insertable_records(
            developer_names,
            original_texts,
        )

        if not original_texts:
            raise ValueError(
                "All uploaded records were skipped because they exceed the current Milvus field length limits. "
                "Use Clear Organization Data to recreate the collection with the updated schema, or shorten those values in the dataset."
            )

        total_records = len(original_texts)
        inserted = 0

        if progress_callback is not None:
            progress_callback(
                "preparing",
                5,
                f"Validated dataset. Preparing {total_records} expertise records. Skipped {skipped_records} records.",
            )

        for start_index in range(0, total_records, self.upload_batch_size):
            end_index = min(start_index + self.upload_batch_size, total_records)
            batch_names = developer_names[start_index:end_index]
            batch_texts = original_texts[start_index:end_index]

            embeddings = self.embedding_service.encode(batch_texts)
            completed_ratio = end_index / total_records

            if progress_callback is not None:
                progress_callback(
                    "embedding",
                    10 + (completed_ratio * 45),
                    f"Created embeddings for {end_index} of {total_records} records.",
                )

            inserted += self.milvus_repository.insert(batch_names, batch_texts, embeddings)

            if progress_callback is not None:
                progress_callback(
                    "storing",
                    55 + (completed_ratio * 40),
                    f"Stored {end_index} of {total_records} vectors in Milvus.",
                )

        if progress_callback is not None:
            progress_callback("finalizing", 98, "Finalizing upload results.")

        return {
            "inserted_records": inserted,
            "inserted_developers": len(set(developer_names)),
            "skipped_records": skipped_records,
            "collection_name": self.settings.milvus_collection_name,
            "total_vectors": self.milvus_repository.count(),
        }

    def clear_organization_data(self) -> dict[str, int | str]:
        deleted_vectors = self.milvus_repository.clear()
        return {
            "deleted_vectors": deleted_vectors,
            "collection_name": self.settings.milvus_collection_name,
            "remaining_vectors": self.milvus_repository.count(),
        }

    def clear_bug_dataset_model(self) -> dict[str, object]:
        return self.classification_service.clear_finetuned_checkpoints()

    def recommend(self, bug_title: str, bug_description: str, k: int) -> dict[str, object]:
        query_text = self._combine_query_text(bug_title, bug_description)
        query_embedding = self.embedding_service.encode_one(query_text)
        search_limit = min(max(k * self.settings.search_limit_multiplier, k), 100)
        raw_hits = self.milvus_repository.search(query_embedding, limit=search_limit)
        model_recommendations = self.classification_service.predict(query_text, top_k=k)

        deduped: dict[str, dict[str, object]] = {}
        for hit in raw_hits:
            developer_name = str(hit["developer_name"])
            similarity_score = float(hit["similarity_score"])
            existing = deduped.get(developer_name)
            if existing is None or similarity_score > float(existing["similarity_score"]):
                deduped[developer_name] = {
                    "developer_name": developer_name,
                    "similarity_score": similarity_score,
                    "matched_bug_text": hit.get("matched_bug_text"),
                    "vector_id": hit.get("vector_id"),
                    "final_score": similarity_score,
                }

        recommendations = sorted(
            deduped.values(),
            key=lambda item: float(item["final_score"]),
            reverse=True,
        )[:k]

        return {
            "query_text": query_text,
            "recommendations": recommendations,
            "model_recommendations": model_recommendations,
            "classifier_enabled": self.classification_service.enabled,
            "active_model_checkpoint": self.classification_service.active_checkpoint_name,
        }

    @staticmethod
    def _combine_query_text(bug_title: str, bug_description: str) -> str:
        return f"Title: {bug_title.strip()}\nDescription: {bug_description.strip()}"

    @staticmethod
    def _expand_history(bug_history: str | list[str]) -> Iterable[str]:
        if isinstance(bug_history, str):
            text = bug_history.strip()
            if text:
                yield text
            return

        for item in bug_history:
            text = item.strip()
            if text:
                yield text

    @staticmethod
    def _extract_record_texts(record: ExpertiseRecord | BugDatasetRecord) -> Iterable[str]:
        if isinstance(record, BugDatasetRecord):
            combined_text = RecommendationService._combine_query_text(record.title, record.description)
            if combined_text.strip():
                yield combined_text
            return

        yield from RecommendationService._expand_history(record.bug_history)

    def _filter_insertable_records(
        self,
        developer_names: list[str],
        original_texts: list[str],
    ) -> tuple[list[str], list[str], int]:
        filtered_names: list[str] = []
        filtered_texts: list[str] = []
        skipped_records = 0

        for developer_name, original_text in zip(developer_names, original_texts, strict=True):
            if len(developer_name) > self.milvus_repository.developer_name_max_length:
                skipped_records += 1
                continue

            if len(original_text) > self.milvus_repository.original_text_max_length:
                skipped_records += 1
                continue

            filtered_names.append(developer_name)
            filtered_texts.append(original_text)

        return filtered_names, filtered_texts, skipped_records
