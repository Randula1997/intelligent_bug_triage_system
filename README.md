# Bug Triage Developer Recommender

This project provides a complete developer recommendation workflow for bug triaging using:

- FastAPI for the backend API
- SentenceTransformers for embedding developer bug-history records and incoming bug queries
- Milvus for vector storage and similarity search
<!-- - A fine-tuned HuggingFace sequence classification checkpoint for optional hybrid reranking -->
- A static HTML, CSS, and JavaScript frontend served by FastAPI

Training and fine-tuning code are intentionally excluded.

## Folder Structure

```text
bug-triage-recommender/
├── app/
│   ├── api/
│   │   └── routes.py
│   ├── core/
│   │   └── config.py
│   ├── db/
│   │   └── milvus.py
│   ├── models/
│   │   └── schemas.py
│   ├── services/
│   │   ├── classification_service.py
│   │   ├── embedding_service.py
│   │   └── recommendation_service.py
│   └── main.py
├── static/
│   ├── app.js
│   ├── index.html
│   └── styles.css
├── .env.example
├── README.md
└── requirements.txt
```

## Core Workflow

### 1. Upload Developer Expertise

Send a dataset where each record contains:

```json
{
  "developer_name": "alice",
  "bug_history": "Crash on workspace reload after extension activation"
}
```

`bug_history` can also be a list of strings if a developer has multiple historical bug records in one row.

The backend will:

1. Parse the uploaded JSON, JSONL, or CSV file.
2. Expand each `bug_history` item into a searchable record.
3. Generate normalized embeddings with the configured SentenceTransformer model.
4. Insert vectors and metadata into Milvus.

Stored metadata includes:

- `developer_name`
- `original_text`
- `embedding vector ID`

### 2. Query Bug Recommendations

The query form sends:

- `bug_title`
- `bug_description`
- `k`

The backend will:

1. Combine title and description into one text block.
2. Generate an embedding with the same embedding model.
3. Search Milvus for similar developer history vectors.
4. Deduplicate matches by developer.
<!-- 5. Optionally blend the vector score with the fine-tuned classifier score when the developer label exists in the checkpoint config. -->

## Setup

### 1. Create and activate a virtual environment

```bash
python -m venv .venv
```

Open a terminal in the project root and activate the environment before installing any dependencies.

On Windows PowerShell:

```bash
.venv\Scripts\Activate.ps1
```

On Windows Command Prompt:

```bash
.venv\Scripts\activate.bat
```

On macOS or Linux:

```bash
source .venv/bin/activate
```

### 2. Install dependencies inside the activated virtual environment

After activation, confirm your shell is using the virtual environment, then install the project requirements:

```bash
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### 3. Install Docker Desktop and start Milvus

Install Docker Desktop for Windows if it is not already installed:

1. Download Docker Desktop from the official Docker website.
2. Install it and complete the initial setup.
3. Launch Docker Desktop and wait until it shows that Docker is running.

Create a separate folder for the Milvus standalone stack:

```powershell
mkdir C:\milvus-docker
```

Inside that folder, create a file named `docker-compose.yml` with the following content:

```yaml
version: "3.5"

services:
  etcd:
    image: quay.io/coreos/etcd:v3.5.5
    container_name: milvus-etcd
    environment:
      - ETCD_AUTO_COMPACTION_MODE=revision
      - ETCD_AUTO_COMPACTION_RETENTION=1000
    command: etcd -advertise-client-urls=http://0.0.0.0:2379 -listen-client-urls=http://0.0.0.0:2379
    ports:
      - "2379:2379"

  minio:
    image: minio/minio:RELEASE.2023-03-20T20-16-18Z
    container_name: milvus-minio
    environment:
      MINIO_ACCESS_KEY: minioadmin
      MINIO_SECRET_KEY: minioadmin
    command: minio server /data
    ports:
      - "9000:9000"

  milvus:
    image: milvusdb/milvus:v2.3.0
    container_name: milvus-standalone
    command: ["milvus", "run", "standalone"]
    environment:
      ETCD_ENDPOINTS: etcd:2379
      MINIO_ADDRESS: minio:9000
    ports:
      - "19530:19530"
      - "9091:9091"
    depends_on:
      - etcd
      - minio
```

Run Milvus:

```powershell
cd C:\milvus-docker
docker compose up -d
```

Milvus will be available at `http://127.0.0.1:19530`.

Verify everything is running:

```powershell
docker ps
```

You should see these containers:

- `milvus-standalone`
- `milvus-etcd`
- `milvus-minio`

### 4. Configure environment variables

```bash
copy .env.example .env
```

<!-- Update `.env` so `CHECKPOINT_PATH` points to your HuggingFace fine-tuned checkpoint directory. -->

If you keep the default Docker Desktop setup, leave `MILVUS_URI=http://127.0.0.1:19530`.

### 5. Run the application

```bash
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

## API Endpoints

### `POST /api/expertise/upload`

Uploads a developer expertise dataset.

Supported file types:

- `.json`
- `.jsonl`
- `.csv`

Example request with `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/api/expertise/upload" \
  -H "accept: application/json" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@sample_expertise.json"
```

Example response:

```json
{
  "inserted_records": 12,
  "inserted_developers": 4,
  "collection_name": "developer_expertise",
  "total_vectors": 12
}
```

### `POST /api/expertise/upload/jobs`

Starts an upload job and returns a job ID that can be polled for progress while embeddings are created and vectors are stored.

Example response:

```json
{
  "job_id": "6fcb5cd5c24e4d14a0e54253a6b5d654",
  "status": "queued",
  "phase": "queued",
  "progress_percent": 0,
  "message": "Upload received. Waiting to create embeddings.",
  "result": null,
  "error": null
}
```

### `GET /api/expertise/upload/jobs/{job_id}`

Returns upload progress for the requested job.

Example response while processing:

```json
{
  "job_id": "6fcb5cd5c24e4d14a0e54253a6b5d654",
  "status": "running",
  "phase": "storing",
  "progress_percent": 78.5,
  "message": "Stored 96 of 128 vectors in Milvus.",
  "result": null,
  "error": null
}
```

### `POST /api/recommend`

Accepts a bug query and returns the top-k developers.

Example request:

```json
{
  "bug_title": "Search panel freezes after refresh",
  "bug_description": "The UI stops responding after refreshing the search panel when a workspace contains symlinked folders.",
  "k": 3
}
```

Example `curl`:

```bash
curl -X POST "http://127.0.0.1:8000/api/recommend" \
  -H "Content-Type: application/json" \
  -d '{
    "bug_title": "Search panel freezes after refresh",
    "bug_description": "The UI stops responding after refreshing the search panel when a workspace contains symlinked folders.",
    "k": 3
  }'
```

Example response:

```json
{
  "query_text": "Title: Search panel freezes after refresh\nDescription: The UI stops responding after refreshing the search panel when a workspace contains symlinked folders.",
  "recommendations": [
    {
      "developer_name": "alice",
      "similarity_score": 0.8734,
      "matched_bug_text": "Search view deadlocks when symbolic links are expanded during refresh.",
      "vector_id": 245613492177,
      "final_score": 0.8487
    },
    {
      "developer_name": "maria",
      "similarity_score": 0.8411,
      "matched_bug_text": "Workspace explorer hangs while invalidating nested file watchers.",
      "vector_id": 245613492178,
      "final_score": 0.7754
    },
    {
      "developer_name": "pranav",
      "similarity_score": 0.8127,
      "matched_bug_text": "Search service degrades after cache invalidation with symlink recursion.",
      "vector_id": 245613492179,
      "final_score": 0.8127
    }
  ]
}
```

<!--
Previous hybrid response fields kept for reference:
"classifier_score": 0.7912
"classifier_predictions": [
  {
    "developer_name": "alice",
    "classifier_score": 0.7912
  }
]
-->

### `DELETE /api/organization-data`

Deletes all vectors stored in the currently configured Milvus collection for the active organization context.

Example `curl`:

```bash
curl -X DELETE "http://127.0.0.1:8000/api/organization-data"
```

Example response:

```json
{
  "deleted_vectors": 12,
  "collection_name": "developer_expertise",
  "remaining_vectors": 0
}
```

### `GET /api/health`

Returns service metadata and collection health.

Example response:

```json
{
  "status": "ok",
  "collection_name": "developer_expertise",
  "vector_count": 12,
  "embedding_model_name": "sentence-transformers/all-MiniLM-L6-v2",
  "classifier_enabled": false
}
```

## Milvus Notes

- The default `MILVUS_URI` points to the local Docker Desktop Milvus server at `http://127.0.0.1:19530`.
- If your Milvus server requires authentication, set `MILVUS_TOKEN` to `username:password` or the token value expected by your deployment.
- If you explicitly want Milvus Lite instead of Docker, point `MILVUS_URI` to a local `.db` path such as `./data/milvus_developer_expertise.db`.
- Vectors are indexed with a `FLAT` index and `IP` metric. Because embeddings are normalized, inner product behaves like cosine similarity.

## Docker Desktop Workflow

1. Start Docker Desktop.
2. From the project root, run `docker compose up -d`.
3. Copy `.env.example` to `.env` if you have not already.
<!-- 4. Set `CHECKPOINT_PATH` in `.env`. -->
4. Start the API with `uvicorn app.main:app --reload`.
5. Open `http://127.0.0.1:8000`.

The first dataset upload creates the `developer_expertise` collection automatically.

## Best Practices Included

- Shared embedding model for both expertise ingestion and bug queries
- Pydantic request and response schemas for API validation
- Separation between API, service layer, and Milvus integration
- Deduplication of vector hits so each developer appears once in the final result list
<!-- - Optional hybrid reranking using classifier scores from the existing checkpoint -->
- Static frontend served from the same FastAPI application for easy deployment

## Sample Expertise Dataset

```json
[
  {
    "developer_name": "alice",
    "bug_history": [
      "Crash on workspace reload after extension activation.",
      "Search panel deadlocks when workspace cache is rebuilt."
    ]
  },
  {
    "developer_name": "maria",
    "bug_history": "File watcher leaks handles when large monorepos are reopened."
  }
]
```
