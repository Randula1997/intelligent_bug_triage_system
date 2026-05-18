from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Bug Triage Developer Recommender"
    api_prefix: str = "/api"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    classifier_base_checkpoint_path: str = Field(default="./checkpoints/model-checkpoint")
    classifier_finetuned_root: str = Field(default="./checkpoints/finetuned-models")
    classifier_train_epochs: int = Field(default=3, ge=1, le=20)
    classifier_learning_rate: float = Field(default=3e-5, gt=0)
    classifier_batch_size: int = Field(default=4, ge=1, le=128)
    classifier_max_length: int = Field(default=256, ge=32, le=1024)
    classifier_gradient_accumulation_steps: int = Field(default=2, ge=1, le=64)
    classifier_early_stopping_patience: int = Field(default=2, ge=1, le=10)
    classifier_mixed_precision_enabled: bool = True
    milvus_uri: str = Field(default="http://127.0.0.1:19530")
    milvus_token: str | None = None
    milvus_collection_name: str = "developer_expertise"
    milvus_index_type: str = "FLAT"
    milvus_metric_type: str = "IP"
    search_limit_multiplier: int = 5
    expertise_upload_batch_size: int = Field(default=256, ge=16, le=2048)
    expertise_chunk_size_tokens: int = Field(default=384, ge=32, le=4096)
    # hybrid_alpha: float = 0.7

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def milvus_uri_path(self) -> Path | None:
        if self.milvus_uri.endswith(".db"):
            return Path(self.milvus_uri)
        return None

    @property
    def classifier_base_checkpoint_dir(self) -> Path:
        return Path(self.classifier_base_checkpoint_path)

    @property
    def classifier_finetuned_root_dir(self) -> Path:
        return Path(self.classifier_finetuned_root)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
