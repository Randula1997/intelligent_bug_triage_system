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


class UploadJobStatusResponse(BaseModel):
    job_id: str
    status: str
    phase: str
    progress_percent: float = Field(ge=0, le=100)
    message: str
    result: UploadResponse | None = None
    error: str | None = None


class ClearOrganizationDataResponse(BaseModel):
    deleted_vectors: int
    collection_name: str
    remaining_vectors: int


class BugQueryRequest(BaseModel):
    bug_title: str = Field(..., min_length=1)
    bug_description: str = Field(..., min_length=1)
    k: int = Field(default=5, ge=1, le=20)


class RecommendationItem(BaseModel):
    developer_name: str
    similarity_score: float
    matched_bug_text: str | None = None
    vector_id: int | None = None
    # classifier_score: float | None = None
    final_score: float


# class ClassifierPrediction(BaseModel):
#     developer_name: str
#     classifier_score: float


class BugQueryResponse(BaseModel):
    query_text: str
    recommendations: list[RecommendationItem]
    # classifier_predictions: list[ClassifierPrediction] = Field(default_factory=list)


class HealthResponse(BaseModel):
    status: str
    collection_name: str
    vector_count: int
    embedding_model_name: str
    classifier_enabled: bool
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
