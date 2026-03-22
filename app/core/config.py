from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    app_name: str = "Bug Triage Developer Recommender"
    api_prefix: str = "/api"
    embedding_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    # checkpoint_path: str = Field(default="./checkpoints/developer-classifier")
    milvus_uri: str = Field(default="http://127.0.0.1:19530")
    milvus_token: str | None = None
    milvus_collection_name: str = "developer_expertise"
    milvus_index_type: str = "FLAT"
    milvus_metric_type: str = "IP"
    search_limit_multiplier: int = 5
    # hybrid_alpha: float = 0.7

    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8", extra="ignore")

    @property
    def milvus_uri_path(self) -> Path | None:
        if self.milvus_uri.endswith(".db"):
            return Path(self.milvus_uri)
        return None


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
