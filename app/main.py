from __future__ import annotations

from contextlib import asynccontextmanager
from pathlib import Path
from threading import Lock

from fastapi import FastAPI
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app.api.routes import router
from app.core.config import get_settings
from app.db.milvus import MilvusRepository
from app.services.classification_service import ClassificationService
from app.services.embedding_service import EmbeddingService
from app.services.recommendation_service import RecommendationService


BASE_DIR = Path(__file__).resolve().parent.parent
STATIC_DIR = BASE_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    app.state.settings = settings
    app.state.recommendation_service = None
    app.state.startup_error = None
    app.state.upload_jobs = {}
    app.state.upload_jobs_lock = Lock()

    try:
        embedding_service = EmbeddingService(settings.embedding_model_name)
        classification_service = ClassificationService(
            base_checkpoint_path=settings.classifier_base_checkpoint_path,
            finetuned_root_path=settings.classifier_finetuned_root,
            train_epochs=settings.classifier_train_epochs,
            learning_rate=settings.classifier_learning_rate,
            batch_size=settings.classifier_batch_size,
            max_length=settings.classifier_max_length,
        )
        milvus_repository = MilvusRepository(
            uri=settings.milvus_uri,
            token=settings.milvus_token,
            collection_name=settings.milvus_collection_name,
            vector_dim=embedding_service.dimension,
            index_type=settings.milvus_index_type,
            metric_type=settings.milvus_metric_type,
        )
        recommendation_service = RecommendationService(
            settings=settings,
            embedding_service=embedding_service,
            classification_service=classification_service,
            milvus_repository=milvus_repository,
        )
        app.state.recommendation_service = recommendation_service
    except Exception as exc:
        app.state.startup_error = str(exc)

    yield


app = FastAPI(title="Bug Triage Developer Recommender", lifespan=lifespan)
app.include_router(router, prefix=get_settings().api_prefix)
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")


@app.get("/", include_in_schema=False)
async def index() -> FileResponse:
    return FileResponse(STATIC_DIR / "index.html")
