from typing import Any

from pydantic import BaseModel, Field, field_validator


class ExpertiseRecord(BaseModel):
    developer_name: str = Field(..., min_length=1)
    bug_history: str | list[str]

    @field_validator("developer_name")
    @classmethod
    def strip_name(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("developer_name cannot be empty")
        return cleaned


class BugDatasetRecord(BaseModel):
    title: str = Field(..., min_length=1)
    description: str = Field(..., min_length=1)
    developer_name: str = Field(..., min_length=1)

    @field_validator("title", "description", "developer_name")
    @classmethod
    def strip_required_text(cls, value: str) -> str:
        cleaned = value.strip()
        if not cleaned:
            raise ValueError("field cannot be empty")
        return cleaned


class UploadResponse(BaseModel):
    inserted_records: int
    inserted_developers: int
    skipped_records: int = 0
    collection_name: str
    total_vectors: int


class BugDatasetUploadResponse(BaseModel):
    accepted_records: int
    developer_count: int
    source_name: str
    required_fields: list[str] = Field(default_factory=lambda: ["title", "description", "developer_name"])


class BugDatasetTrainingResult(BaseModel):
    source_name: str
    trained_records: int
    developer_count: int
    base_checkpoint_path: str
    output_checkpoint_path: str
    epochs: int
    learning_rate: float
    batch_size: int
    max_length: int
    created_at: str


class UploadJobStatusResponse(BaseModel):
    job_id: str
    status: str
    phase: str
    progress_percent: float = Field(ge=0, le=100)
    message: str
    result: UploadResponse | None = None
    error: str | None = None


class TrainingJobStatusResponse(BaseModel):
    job_id: str
    status: str
    phase: str
    progress_percent: float = Field(ge=0, le=100)
    message: str
    result: BugDatasetTrainingResult | None = None
    error: str | None = None


class ClearOrganizationDataResponse(BaseModel):
    deleted_vectors: int
    collection_name: str
    remaining_vectors: int


class ClearBugDatasetModelResponse(BaseModel):
    deleted_checkpoints: int
    deleted_checkpoint_paths: list[str] = Field(default_factory=list)
    base_checkpoint_path: str
    active_checkpoint_path: str | None = None


class BugQueryRequest(BaseModel):
    bug_title: str = Field(..., min_length=1)
    bug_description: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class RecommendationItem(BaseModel):
    developer_name: str
    similarity_score: float
    matched_bug_text: str | None = None
    vector_id: int | None = None
    final_score: float


class ModelRecommendationItem(BaseModel):
    developer_name: str
    model_score: float


class BugQueryResponse(BaseModel):
    query_text: str
    recommendations: list[RecommendationItem]
    model_recommendations: list[ModelRecommendationItem] = Field(default_factory=list)
    classifier_enabled: bool = False
    active_model_checkpoint: str | None = None


class HealthResponse(BaseModel):
    status: str
    collection_name: str
    vector_count: int
    embedding_model_name: str
    classifier_enabled: bool
    classifier_base_checkpoint: str | None = None
    classifier_active_checkpoint: str | None = None
    startup_error: str | None = None


class SearchHit(BaseModel):
    developer_name: str
    similarity_score: float
    matched_bug_text: str | None = None
    vector_id: int | None = None


class ParsedDataset(BaseModel):
    records: list[ExpertiseRecord]
    source_name: str


class ErrorMessage(BaseModel):
    detail: str | dict[str, Any] | list[Any]
