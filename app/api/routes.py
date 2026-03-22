from __future__ import annotations

import csv
import io
import json
from pathlib import Path
from typing import TypeVar
from uuid import uuid4

from fastapi import APIRouter, BackgroundTasks, File, HTTPException, Request, UploadFile, status
from pydantic import BaseModel, ValidationError

from app.models.schemas import (
    BugDatasetRecord,
    BugDatasetUploadResponse,
    BugQueryRequest,
    BugQueryResponse,
    ClearOrganizationDataResponse,
    ExpertiseRecord,
    HealthResponse,
    UploadJobStatusResponse,
    UploadResponse,
)


router = APIRouter()
RecordModelT = TypeVar("RecordModelT", bound=BaseModel)


def _set_upload_job(request: Request, job_id: str, payload: dict[str, object]) -> None:
    with request.app.state.upload_jobs_lock:
        existing = request.app.state.upload_jobs.get(job_id, {})
        request.app.state.upload_jobs[job_id] = {**existing, **payload}


def _get_upload_job(request: Request, job_id: str) -> dict[str, object]:
    with request.app.state.upload_jobs_lock:
        payload = request.app.state.upload_jobs.get(job_id)

    if payload is None:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Upload job was not found.")

    return payload


def _get_service_or_raise(request: Request):
    service = getattr(request.app.state, "recommendation_service", None)
    if service is None:
        startup_error = getattr(request.app.state, "startup_error", "Service initialization failed.")
        raise HTTPException(status_code=status.HTTP_503_SERVICE_UNAVAILABLE, detail=startup_error)
    return service


@router.get("/health", response_model=HealthResponse)
async def health(request: Request) -> HealthResponse:
    service = getattr(request.app.state, "recommendation_service", None)
    settings = request.app.state.settings

    if service is None:
        return HealthResponse(
            status="degraded",
            collection_name=settings.milvus_collection_name,
            vector_count=0,
            embedding_model_name=settings.embedding_model_name,
            classifier_enabled=False,
            startup_error=getattr(request.app.state, "startup_error", None),
        )

    return HealthResponse(
        status="ok",
        collection_name=settings.milvus_collection_name,
        vector_count=service.milvus_repository.count(),
        embedding_model_name=settings.embedding_model_name,
        classifier_enabled=False,
        # classifier_enabled=service.classification_service.enabled,
        startup_error=None,
    )


@router.post("/expertise/upload", response_model=UploadResponse)
async def upload_expertise(request: Request, file: UploadFile = File(...)) -> UploadResponse:
    records = await _parse_typed_dataset(file, ExpertiseRecord)
    service = _get_service_or_raise(request)

    try:
        result = service.upload_expertise(records)
    except ValueError as exc:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail=str(exc)) from exc

    return UploadResponse(**result)


@router.post(
    "/expertise/upload/jobs",
    response_model=UploadJobStatusResponse,
    status_code=status.HTTP_202_ACCEPTED,
)
async def create_upload_job(
    request: Request,
    background_tasks: BackgroundTasks,
    file: UploadFile = File(...),
) -> UploadJobStatusResponse:
    service = _get_service_or_raise(request)
    records = await _parse_typed_dataset(file, ExpertiseRecord)
    job_id = uuid4().hex

    _set_upload_job(
        request,
        job_id,
        {
            "job_id": job_id,
            "status": "queued",
            "phase": "queued",
            "progress_percent": 0.0,
            "message": "Upload received. Waiting to create embeddings.",
            "result": None,
            "error": None,
        },
    )

    background_tasks.add_task(_process_upload_job, request, job_id, service, records)
    return UploadJobStatusResponse(**_get_upload_job(request, job_id))


@router.get("/expertise/upload/jobs/{job_id}", response_model=UploadJobStatusResponse)
async def get_upload_job_status(request: Request, job_id: str) -> UploadJobStatusResponse:
    return UploadJobStatusResponse(**_get_upload_job(request, job_id))


@router.post("/bug-dataset/upload", response_model=BugDatasetUploadResponse)
async def upload_bug_dataset(file: UploadFile = File(...)) -> BugDatasetUploadResponse:
    records = await _parse_typed_dataset(file, BugDatasetRecord)
    return BugDatasetUploadResponse(
        accepted_records=len(records),
        developer_count=len({record.developer_name for record in records}),
        source_name=file.filename or "uploaded_bug_dataset",
    )


@router.post("/recommend", response_model=BugQueryResponse)
async def recommend(request: Request, payload: BugQueryRequest) -> BugQueryResponse:
    service = _get_service_or_raise(request)
    result = service.recommend(payload.bug_title, payload.bug_description, payload.k)
    return BugQueryResponse(**result)


@router.delete("/organization-data", response_model=ClearOrganizationDataResponse)
async def clear_organization_data(request: Request) -> ClearOrganizationDataResponse:
    service = _get_service_or_raise(request)
    result = service.clear_organization_data()
    return ClearOrganizationDataResponse(**result)


async def _parse_typed_dataset(file: UploadFile, model_type: type[RecordModelT]) -> list[RecordModelT]:
    filename = file.filename or "uploaded_dataset"
    suffix = Path(filename).suffix.lower()
    raw_bytes = await file.read()

    if not raw_bytes:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded file is empty.")

    try:
        decoded = raw_bytes.decode("utf-8-sig")
    except UnicodeDecodeError as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Only UTF-8 encoded files are supported.",
        ) from exc

    try:
        if suffix == ".json":
            parsed = json.loads(decoded)
            if not isinstance(parsed, list):
                raise HTTPException(
                    status_code=status.HTTP_400_BAD_REQUEST,
                    detail="JSON files must contain an array of records.",
                )
            return _validate_typed_records(parsed, model_type)

        if suffix == ".jsonl":
            parsed_lines = [json.loads(line) for line in decoded.splitlines() if line.strip()]
            return _validate_typed_records(parsed_lines, model_type)

        if suffix == ".csv":
            reader = csv.DictReader(io.StringIO(decoded))
            rows = [row for row in reader if not _is_empty_row(row)]
            return _validate_typed_records(rows, model_type)
    except HTTPException:
        raise
    except (json.JSONDecodeError, ValueError, ValidationError) as exc:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=f"Could not parse uploaded dataset: {exc}",
        ) from exc

    raise HTTPException(
        status_code=status.HTTP_400_BAD_REQUEST,
        detail="Unsupported file format. Use .json, .jsonl, or .csv.",
    )


def _validate_typed_records(items: list[object], model_type: type[RecordModelT]) -> list[RecordModelT]:
    records: list[RecordModelT] = []

    for index, item in enumerate(items, start=1):
        if _is_empty_item(item):
            continue

        try:
            records.append(model_type.model_validate(item))
        except ValidationError as exc:
            raise ValueError(f"record {index}: {exc}") from exc

    return records


def _is_empty_row(row: dict[str, object]) -> bool:
    return all(not str(value or "").strip() for value in row.values())


def _is_empty_item(item: object) -> bool:
    if isinstance(item, dict):
        return all(not str(value or "").strip() for value in item.values())

    return False


def _process_upload_job(
    request: Request,
    job_id: str,
    service,
    records: list[ExpertiseRecord],
) -> None:
    _set_upload_job(
        request,
        job_id,
        {
            "status": "running",
            "phase": "preparing",
            "progress_percent": 1.0,
            "message": "Preparing uploaded expertise data.",
            "result": None,
            "error": None,
        },
    )

    try:
        result = service.upload_expertise(
            records,
            progress_callback=lambda phase, percent, message: _set_upload_job(
                request,
                job_id,
                {
                    "status": "running",
                    "phase": phase,
                    "progress_percent": round(percent, 2),
                    "message": message,
                },
            ),
        )
    except Exception as exc:
        _set_upload_job(
            request,
            job_id,
            {
                "status": "failed",
                "phase": "failed",
                "progress_percent": 100.0,
                "message": "Expertise upload failed.",
                "error": str(exc),
                "result": None,
            },
        )
        return

    _set_upload_job(
        request,
        job_id,
        {
            "status": "completed",
            "phase": "completed",
            "progress_percent": 100.0,
            "message": "Expertise upload completed successfully.",
            "result": result,
            "error": None,
        },
    )
