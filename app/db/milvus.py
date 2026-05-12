from __future__ import annotations

from pathlib import Path

from pymilvus import Collection, CollectionSchema, DataType, FieldSchema, MilvusException, connections, utility


class MilvusRepository:
    default_developer_name_max_length = 65535
    default_original_text_max_length = 65535

    def __init__(
        self,
        *,
        uri: str,
        token: str | None,
        collection_name: str,
        vector_dim: int,
        index_type: str,
        metric_type: str,
    ) -> None:
        self.uri = uri
        self.token = token
        self.collection_name = collection_name
        self.vector_dim = vector_dim
        self.index_type = index_type
        self.metric_type = metric_type
        self.alias = "default"
        self.developer_name_max_length = self.default_developer_name_max_length
        self.original_text_max_length = self.default_original_text_max_length

        if self._is_local_db_uri(uri):
            Path(uri).parent.mkdir(parents=True, exist_ok=True)

        connection_kwargs = {"alias": self.alias, "uri": uri}
        if token:
            connection_kwargs["token"] = token

        try:
            connections.connect(**connection_kwargs)
            self.collection = self._ensure_collection()
        except MilvusException as exc:
            raise RuntimeError(self._build_connection_error(str(exc))) from exc

    @staticmethod
    def _is_local_db_uri(uri: str) -> bool:
        return uri.endswith(".db")

    def _build_connection_error(self, reason: str) -> str:
        if self._is_local_db_uri(self.uri):
            return f"Could not open Milvus Lite database at '{self.uri}': {reason}"

        return (
            "Could not connect to the Milvus server at "
            f"'{self.uri}'. Start the Docker Desktop stack with 'docker compose up -d' "
            "and confirm port 19530 is reachable before launching the API. "
            f"Milvus error: {reason}"
        )

    def _ensure_collection(self) -> Collection:
        if utility.has_collection(self.collection_name, using=self.alias):
            collection = Collection(name=self.collection_name, using=self.alias)
            self._sync_field_limits(collection)
            collection.load()
            return collection

        schema = CollectionSchema(
            fields=[
                FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
                FieldSchema(
                    name="developer_name",
                    dtype=DataType.VARCHAR,
                    max_length=self.default_developer_name_max_length,
                ),
                FieldSchema(
                    name="original_text",
                    dtype=DataType.VARCHAR,
                    max_length=self.default_original_text_max_length,
                ),
                FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=self.vector_dim),
            ],
            description="Developer expertise embeddings for bug triage",
        )
        collection = Collection(name=self.collection_name, schema=schema, using=self.alias)
        self._sync_field_limits(collection)
        collection.create_index(
            field_name="embedding",
            index_params={
                "index_type": self.index_type,
                "metric_type": self.metric_type,
                "params": {},
            },
        )
        collection.load()
        return collection

    def _sync_field_limits(self, collection: Collection) -> None:
        for field in collection.schema.fields:
            if field.name == "developer_name":
                self.developer_name_max_length = self._get_field_max_length(
                    field,
                    self.default_developer_name_max_length,
                )
            elif field.name == "original_text":
                self.original_text_max_length = self._get_field_max_length(
                    field,
                    self.default_original_text_max_length,
                )

    @staticmethod
    def _get_field_max_length(field: FieldSchema, default: int) -> int:
        params = getattr(field, "params", None)
        if isinstance(params, dict) and "max_length" in params:
            return int(params["max_length"])

        max_length = getattr(field, "max_length", None)
        if max_length is not None:
            return int(max_length)

        return default

    def _validate_string_lengths(self, developer_names: list[str], original_texts: list[str]) -> None:
        for index, developer_name in enumerate(developer_names, start=1):
            if len(developer_name) > self.developer_name_max_length:
                raise ValueError(
                    "Could not store uploaded dataset because record "
                    f"{index} has developer_name length {len(developer_name)}, "
                    f"which exceeds the current Milvus collection limit of {self.developer_name_max_length}. "
                    "If this collection was created earlier with a smaller schema, use Clear Developer Data "
                    "to recreate it with the updated limit, or shorten that developer_name value in the dataset."
                )

        for index, original_text in enumerate(original_texts, start=1):
            if len(original_text) > self.original_text_max_length:
                raise ValueError(
                    "Could not store uploaded dataset because record "
                    f"{index} has bug history text length {len(original_text)}, "
                    f"which exceeds the current Milvus collection limit of {self.original_text_max_length}."
                )

    def insert(self, developer_names: list[str], original_texts: list[str], embeddings: list[list[float]]) -> int:
        self._validate_string_lengths(developer_names, original_texts)
        insert_result = self.collection.insert([developer_names, original_texts, embeddings])
        self.collection.flush()
        self.collection.load()
        return len(insert_result.primary_keys)

    def clear(self) -> int:
        existing_count = self.count()
        if utility.has_collection(self.collection_name, using=self.alias):
            self.collection.release()
            utility.drop_collection(self.collection_name, using=self.alias)

        self.collection = self._ensure_collection()
        return existing_count

    def search(self, embedding: list[float], limit: int) -> list[dict[str, str | float | int | None]]:
        search_result = self.collection.search(
            data=[embedding],
            anns_field="embedding",
            param={"metric_type": self.metric_type, "params": {}},
            limit=limit,
            output_fields=["developer_name", "original_text"],
        )

        hits: list[dict[str, str | float | int | None]] = []
        for match in search_result[0]:
            entity = match.entity
            hits.append(
                {
                    "vector_id": int(match.id),
                    "developer_name": entity.get("developer_name"),
                    "matched_bug_text": entity.get("original_text"),
                    "similarity_score": float(match.score),
                }
            )
        return hits

    def count(self) -> int:
        return int(self.collection.num_entities)
