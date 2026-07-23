<div align="center">

# 📧 Production MailGuard API

### A production-grade, MLOps-driven Spam Email Classification service

Built with **FastAPI**, versioned with **DVC**, tracked with **MLflow / DagsHub**, and served through a reproducible **NLP + TF‑IDF + Logistic Regression** pipeline.

<br/>

[![Live Demo](https://img.shields.io/badge/Live_Demo-54.83.143.31%3A8000-success?style=for-the-badge&logo=amazonaws&logoColor=white)](http://54.83.143.31:8000/docs)

[![Python](https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://www.python.org/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.139-009688?style=for-the-badge&logo=fastapi&logoColor=white)](https://fastapi.tiangolo.com/)
[![scikit-learn](https://img.shields.io/badge/scikit--learn-1.9-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)](https://scikit-learn.org/)
[![spaCy](https://img.shields.io/badge/spaCy-3.8-09A3D5?style=for-the-badge&logo=spacy&logoColor=white)](https://spacy.io/)
[![MLflow](https://img.shields.io/badge/MLflow-3.14-0194E2?style=for-the-badge&logo=mlflow&logoColor=white)](https://mlflow.org/)
[![DVC](https://img.shields.io/badge/DVC-3.67-945DD6?style=for-the-badge&logo=dvc&logoColor=white)](https://dvc.org/)
[![DagsHub](https://img.shields.io/badge/DagsHub-Model_Registry-FF6600?style=for-the-badge)](https://dagshub.com/)
[![AWS S3](https://img.shields.io/badge/AWS_S3-DVC_Remote_%26_Config-232F3E?style=for-the-badge&logo=amazons3&logoColor=white)](https://aws.amazon.com/s3/)
[![AWS ECR](https://img.shields.io/badge/AWS_ECR-Image_Registry-232F3E?style=for-the-badge&logo=amazonecs&logoColor=white)](https://aws.amazon.com/ecr/)
[![AWS EC2](https://img.shields.io/badge/AWS_EC2-Deployed-232F3E?style=for-the-badge&logo=amazonec2&logoColor=white)](https://aws.amazon.com/ec2/)
[![Docker](https://img.shields.io/badge/Docker-Containerized-2496ED?style=for-the-badge&logo=docker&logoColor=white)](https://www.docker.com/)
[![uv](https://img.shields.io/badge/uv-package_manager-DE5FE9?style=for-the-badge&logo=uv&logoColor=white)](https://docs.astral.sh/uv/)
[![Pandas](https://img.shields.io/badge/Pandas-2.3-150458?style=for-the-badge&logo=pandas&logoColor=white)](https://pandas.pydata.org/)
[![Pydantic](https://img.shields.io/badge/Pydantic-Validated-E92063?style=for-the-badge&logo=pydantic&logoColor=white)](https://docs.pydantic.dev/)
[![Ruff](https://img.shields.io/badge/Linter-Ruff-D7FF64?style=for-the-badge&logo=ruff&logoColor=black)](https://docs.astral.sh/ruff/)

[![License](https://img.shields.io/badge/License-Unlicensed-lightgrey?style=for-the-badge)](#-license)
[![Repo Size](https://img.shields.io/github/repo-size/Sharanch3/Production-MailGuard-API?style=for-the-badge&color=blue)](https://github.com/Sharanch3/Production-MailGuard-API)
[![Last Commit](https://img.shields.io/github/last-commit/Sharanch3/Production-MailGuard-API?style=for-the-badge&color=orange)](https://github.com/Sharanch3/Production-MailGuard-API/commits/main)
[![Issues](https://img.shields.io/github/issues/Sharanch3/Production-MailGuard-API?style=for-the-badge&color=red)](https://github.com/Sharanch3/Production-MailGuard-API/issues)
[![Stars](https://img.shields.io/github/stars/Sharanch3/Production-MailGuard-API?style=for-the-badge&color=yellow)](https://github.com/Sharanch3/Production-MailGuard-API/stargazers)

</div>

---

## 📖 Table of Contents

- [Overview](#-overview)
- [Live Demo](#-live-demo)
- [Architecture](#-architecture)
- [Tech Stack](#-tech-stack)
- [Project Structure](#-project-structure)
- [ML Pipeline (DVC)](#-ml-pipeline-dvc)
- [Getting Started](#-getting-started)
  - [Prerequisites](#prerequisites)
  - [Installation](#installation)
  - [Environment Variables](#environment-variables)
  - [Reproducing the Pipeline](#reproducing-the-pipeline)
  - [Running the API](#running-the-api)
- [Running with Docker](#-running-with-docker)
- [Production Deployment (AWS EC2 + ECR)](#-production-deployment-aws-ec2--ecr)
- [API Reference](#-api-reference)
- [Model Details](#-model-details)
- [Configuration](#-configuration)
- [Experiment Tracking](#-experiment-tracking)
- [Request Auditing](#-request-auditing)
- [Logging](#-logging)
- [Roadmap](#-roadmap)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🧠 Overview

**Production MailGuard API** is an end-to-end, production-style **Spam vs. Ham email classifier**. It goes beyond a simple notebook model — the project is structured as a complete **MLOps pipeline**:

1. **Data Ingestion** — pulls the raw email dataset from a remote CSV source and performs a train/test split.
2. **Data Preprocessing** — cleans and normalizes raw email text (URL/HTML/email removal, lemmatization) using a **spaCy** NLP pipeline.
3. **Feature Engineering** — vectorizes text with **TF-IDF** and persists the fitted vectorizer as an artifact.
4. **Model Building** — trains a **Logistic Regression** classifier on the TF-IDF features.
5. **Model Evaluation & Registry** — computes accuracy/precision/recall/AUC and logs the run (metrics, params, model) to **MLflow**, registering the model in the **DagsHub Model Registry**.
6. **Serving** — a **FastAPI** application loads the registered model, the TF-IDF vectorizer, and the spaCy pipeline at startup and exposes a REST API for real-time spam classification.
7. **Auditing** — every `/predict` call is appended to a persisted `audit/emails.csv` file (input text + predicted label), giving a lightweight, inspectable trail of what the model has classified in production.
8. **Containerization** — the API ships with a **Dockerfile** and **Docker Compose** setup, so the whole service can be built and run as a single container with the audit log bind-mounted to the host.
9. **Deployment** — the container image is pushed to **AWS ECR** and pulled/run on an **AWS EC2** instance via `docker compose`, making the API publicly reachable and continuously collecting live audit data.

Every pipeline stage is version-controlled and reproducible via **DVC**, with raw/interim/processed data and model artifacts tracked and stored on an **AWS S3** remote.

---

## 🌐 Live Demo

The API is deployed and publicly reachable on an **AWS EC2** instance:

- **Base URL:** [`http://54.83.143.31:8000`](http://54.83.143.31:8000)
- **Interactive Swagger docs:** [`http://54.83.143.31:8000/docs`](http://54.83.143.31:8000/docs)
- **Health check:** [`http://54.83.143.31:8000/health`](http://54.83.143.31:8000/health)

```bash
curl -X POST http://54.83.143.31:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "Congratulations! You have won a $1000 gift card, click here to claim now!!!"}'
```

> ⚠️ This points at a specific EC2 instance's public IPv4 address, which is **not** an Elastic IP — it can change if the instance is stopped/restarted. If the link above is unreachable, the instance may have been stopped or reassigned a new address.

Every request made against the live demo is logged to the server's `audit/emails.csv` (see [Request Auditing](#-request-auditing)), so the live deployment is also continuously accumulating real inference data.

---

## 🏗 Architecture

```
                         ┌─────────────────────────────────────────────────────────┐
                         │                    DVC PIPELINE (dvc.yaml)               │
                         │                                                          │
   emails.csv (remote)   │   ┌──────────────┐   ┌────────────────┐   ┌───────────┐  │
  ───────────────────────┼──▶│ data-ingestion│──▶│data_preprocessing│─▶│  feature-  │ │
                         │   │ (train/test  │   │ (spaCy clean +  │   │ engineering│  │
                         │   │   split)     │   │  lemmatize)     │   │  (TF-IDF)  │  │
                         │   └──────────────┘   └────────────────┘   └─────┬─────┘  │
                         │                                                  │        │
                         │                                                  ▼        │
                         │   ┌──────────────┐   ┌────────────────┐   vectorizer.joblib│
                         │   │model-evaluation◀──│ model_building │◀──────────────────┘
                         │   │ (metrics +   │   │ (Logistic      │                    │
                         │   │  MLflow log) │   │  Regression)   │                    │
                         │   └──────┬───────┘   └────────────────┘                    │
                         └──────────┼───────────────────────────────────────────────┘
                                    │  logs metrics, params & model
                                    ▼
                         ┌─────────────────────────┐
                         │  MLflow / DagsHub        │
                         │  Model Registry          │
                         │  "MailGuard-API" (v1)    │
                         └───────────┬─────────────┘
                                     │ mlflow.sklearn.load_model()
                                     ▼
                         ┌─────────────────────────────────────────┐
                         │             FastAPI Service              │
                         │  ┌───────────┐ ┌───────────┐ ┌─────────┐ │
                         │  │  Model    │ │ Vectorizer│ │  spaCy   │ │
                         │  │ (Log Reg) │ │ (TF-IDF)  │ │   NLP    │ │
                         │  └───────────┘ └───────────┘ └─────────┘ │
                         │        loaded once at app startup         │
                         │                                            │
                         │  GET /          GET /health                │
                         │  POST /predict  GET /model/info            │
                         └───────────────────┬───────────────────────┘
                                              │  JSON
                                              ▼
                                     Client / Consumer App

                         ┌─────────────────────────────────────────────────────────┐
                         │                 DEPLOYMENT (AWS)                        │
                         │                                                         │
                         │   docker build ──▶ AWS ECR (image registry)             │
                         │                         │                               │
                         │                         │ docker pull                   │
                         │                         ▼                               │
                         │              AWS EC2 instance                           │
                         │   ┌─────────────────────────────────────────────────┐   │
                         │   │  compose.yaml + .env  (fetched from AWS S3)      │   │
                         │   │        │                                        │   │
                         │   │        ▼                                        │   │
                         │   │  docker compose up  →  mailguard container       │   │
                         │   │        │                    │                   │   │
                         │   │        │           bind mount ./audit           │   │
                         │   │        ▼                    ▼                   │   │
                         │   │  :8000 exposed        audit/emails.csv           │   │
                         │   │  (public IP)         (persists on host)          │   │
                         │   └─────────────────────────────────────────────────┘   │
                         └─────────────────────────────────────────────────────────┘
                                              │
                                              ▼
                          Live: http://54.83.143.31:8000
```

Raw, interim, and processed datasets — along with the trained model and vectorizer — are all tracked by **DVC** and pushed to an **S3 remote (`s3://sharanch-dvc-bucket`)**, keeping the Git repository lightweight while preserving full data/model lineage. **AWS S3** is now also used to store the deployment-time `compose.yaml` and `.env` files, which are pulled onto the EC2 instance's working directory before `docker compose up` is run.

---

## 🧰 Tech Stack

| Category | Tool | Purpose |
|---|---|---|
| ![Python](https://img.shields.io/badge/-Python-3776AB?style=flat-square&logo=python&logoColor=white) | **Python 3.13** | Core language |
| ![FastAPI](https://img.shields.io/badge/-FastAPI-009688?style=flat-square&logo=fastapi&logoColor=white) | **FastAPI + Uvicorn** | REST API framework & ASGI server |
| ![Pydantic](https://img.shields.io/badge/-Pydantic-E92063?style=flat-square&logo=pydantic&logoColor=white) | **Pydantic** | Request/response schema validation |
| ![scikit-learn](https://img.shields.io/badge/-scikit--learn-F7931E?style=flat-square&logo=scikitlearn&logoColor=white) | **scikit-learn** | TF-IDF vectorization & Logistic Regression model |
| ![spaCy](https://img.shields.io/badge/-spaCy-09A3D5?style=flat-square&logo=spacy&logoColor=white) | **spaCy (`en_core_web_sm`)** | Text cleaning, lemmatization, stopword removal |
| ![Pandas](https://img.shields.io/badge/-Pandas-150458?style=flat-square&logo=pandas&logoColor=white) | **Pandas** | Data manipulation |
| ![DVC](https://img.shields.io/badge/-DVC-945DD6?style=flat-square&logo=dvc&logoColor=white) | **DVC** | Data & pipeline versioning (`dvc.yaml`, `dvc.lock`) |
| ![AWS S3](https://img.shields.io/badge/-AWS_S3-232F3E?style=flat-square&logo=amazons3&logoColor=white) | **AWS S3** | Remote storage for DVC artifacts, and for deployment config (`compose.yaml`, `.env`) pulled onto EC2 |
| ![MLflow](https://img.shields.io/badge/-MLflow-0194E2?style=flat-square&logo=mlflow&logoColor=white) | **MLflow** | Experiment tracking, metric/param logging, model registry |
| ![DagsHub](https://img.shields.io/badge/-DagsHub-FF6600?style=flat-square) | **DagsHub** | Hosted MLflow tracking server & model registry backend |
| ![Docker](https://img.shields.io/badge/-Docker-2496ED?style=flat-square&logo=docker&logoColor=white) | **Docker / Docker Compose** | Containerized build & runtime (`Dockerfile`, `compose.yaml`) |
| ![AWS ECR](https://img.shields.io/badge/-AWS_ECR-232F3E?style=flat-square&logo=amazonecs&logoColor=white) | **AWS ECR** | Private registry hosting the built `mailguard` image |
| ![AWS EC2](https://img.shields.io/badge/-AWS_EC2-232F3E?style=flat-square&logo=amazonec2&logoColor=white) | **AWS EC2** | Compute instance running the deployed container |
| ![uv](https://img.shields.io/badge/-uv-DE5FE9?style=flat-square&logo=uv&logoColor=white) | **uv** | Fast Python package & environment manager (`uv.lock`) |
| ![Ruff](https://img.shields.io/badge/-Ruff-D7FF64?style=flat-square&logo=ruff&logoColor=black) | **Ruff** | Linting & code formatting |
| ![dotenv](https://img.shields.io/badge/-python--dotenv-ECD53F?style=flat-square&logo=python&logoColor=black) | **python-dotenv** | Environment variable / secrets management |
| ![YAML](https://img.shields.io/badge/-YAML-CB171E?style=flat-square&logo=yaml&logoColor=white) | **YAML** (`params.yaml`) | Centralized pipeline hyperparameters |
| ![Joblib](https://img.shields.io/badge/-Joblib-4B8BBE?style=flat-square) | **Joblib** | Model & vectorizer serialization |

---

## 📂 Project Structure

```
Production-MailGuard-API/
├── app/                          # FastAPI application
│   ├── app.py                    # API routes & lifespan (model loading, audit logging)
│   ├── schemas.py                # Pydantic request/response models
│   └── utility.py                # Model/vectorizer/NLP loading + preprocessing
│
├── src/                          # DVC pipeline stages (training code)
│   ├── data_ingestion.py         # Download data, train/test split
│   ├── data_preprocessing.py     # Text cleaning & normalization
│   ├── feature_engineering.py    # TF-IDF vectorization
│   ├── model_building.py         # Logistic Regression training
│   ├── model_evaluation.py       # Metrics + MLflow/DagsHub model registration
│   └── utils.py                  # Shared helpers (logger, params loader, NLP)
│
├── data/
│   ├── raw/                      # train.csv / test.csv (DVC-tracked)
│   ├── interim/                  # Cleaned/preprocessed text (DVC-tracked)
│   └── processed/                # TF-IDF feature matrices (DVC-tracked)
│
├── artifacts/                    # vectorizer.joblib, model.joblib (DVC-tracked)
├── audit/                        # emails.csv — logged prediction requests (bind-mounted in Docker)
├── reports/                      # metrics.json (evaluation output)
├── logs/                         # Per-stage rotating log files
├── expirements/                  # Notebook/experiment scratch space
│
├── Dockerfile                    # Container image definition for the API
├── compose.yaml                  # Docker Compose service definition (build, ports, audit volume)
├── dvc.yaml                      # DVC pipeline stage definitions
├── dvc.lock                      # DVC pipeline lockfile (hashes, deps, outs)
├── params.yaml                   # Centralized hyperparameters
├── .dvc/config                   # DVC remote config (S3 bucket)
├── pyproject.toml                # Project metadata & dependencies (uv)
├── uv.lock                       # Locked dependency versions
└── .python-version               # Python 3.13
```

---

## 🔄 ML Pipeline (DVC)

The full training pipeline is orchestrated with `dvc.yaml` and can be reproduced end-to-end with a single command.

| Stage | Script | Depends On | Params Used | Outputs |
|---|---|---|---|---|
| `data-ingestion` | `src/data_ingestion.py` | — | `data-ingestion.test_size` | `data/raw/` |
| `data_preprocessing` | `src/data_preprocessing.py` | `data/raw/` | — | `data/interim/{train,test}_processed.csv` |
| `feature-engineering` | `src/feature_engineering.py` | `data/interim/*` | `feature-engineering.max_features` | `data/processed/*`, `artifacts/vectorizer.joblib` |
| `model_building` | `src/model_building.py` | `data/processed/*` | `model-building.{solver,penalty,random_state}` | `artifacts/model.joblib` |
| `model-evaluation` | `src/model_evaluation.py` | `artifacts/model.joblib` | — | `reports/metrics.json` + MLflow run |

Run the whole pipeline:

```bash
dvc repro
```

Visualize the DAG:

```bash
dvc dag
```

Pull versioned data/artifacts from the S3 remote:

```bash
dvc pull
```

---

## 🚀 Getting Started

### Prerequisites

- Python **3.13**
- [`uv`](https://docs.astral.sh/uv/) package manager
- AWS credentials configured (for `dvc pull`/`dvc push` against the S3 remote)
- A [DagsHub](https://dagshub.com/) account + access token (for MLflow tracking & model registry)
- [Docker](https://www.docker.com/) + Docker Compose (optional, only needed for containerized deployment)

### Installation

```bash
# Clone the repository
git clone https://github.com/Sharanch3/Production-MailGuard-API.git
cd Production-MailGuard-API

# Install dependencies (creates a .venv automatically)
uv sync

# Activate the environment
# Windows:
.venv\Scripts\activate
# macOS / Linux:
source .venv/bin/activate
```

> The `en_core_web_sm` spaCy model is declared directly in `pyproject.toml` as a source URL, so `uv sync` installs it automatically — no separate `spacy download` step required.

### Environment Variables

Create a `.env` file in the project root:

```env
DAGSHUB_PAT=your_dagshub_access_token
```

This token is used to authenticate with DagsHub for MLflow experiment tracking and model registry access (`src/model_evaluation.py`, `app/utility.py`). The same `.env` file is reused by `compose.yaml` (via `env_file`) when running the API in Docker.

### Reproducing the Pipeline

```bash
dvc repro
```

This runs ingestion → preprocessing → feature engineering → model building → evaluation, and registers the resulting model to the DagsHub-hosted MLflow **Model Registry** under the name `MailGuard-API`.

### Running the API

Run from the **project root** (the `app/` module now uses package-relative imports — `from app.schemas import ...` / `from app.utility import ...` — so it must be launched as part of the `app` package, not as a standalone script inside `app/`):

```bash
uvicorn app.app:app --host 127.0.0.1 --port 8000 --reload
```

The API will be available at `http://127.0.0.1:8000`, with interactive Swagger docs at `http://127.0.0.1:8000/docs`.

On startup, the app loads three resources once into memory (`app.state`):
- ✅ Trained Logistic Regression model (pulled from the MLflow Model Registry)
- ✅ Fitted TF-IDF vectorizer (`artifacts/vectorizer.joblib`)
- ✅ spaCy NLP pipeline (`en_core_web_sm`, parser/NER disabled for speed)

---

## 🐳 Running with Docker

The API is packaged as a Docker image using **uv** for fast, reproducible dependency installation. The current `compose.yaml` in this repo is configured for **production** — it pulls the image from **AWS ECR** rather than building locally (see [Production Deployment](#-production-deployment-aws-ec2--ecr) for how that image gets there).

### Run the published image with Docker Compose

```bash
docker compose up
```

`compose.yaml`:
```yaml
services:
  mailguard:
    image: 707578706440.dkr.ecr.us-east-1.amazonaws.com/sharanch33/mailguard:latest
    container_name: mailguard
    env_file:
      - .env
    ports:
      - "8000:8000"
    volumes:
      - type: bind
        source: ./audit/
        target: /pmg-api/audit/
```

This:
- Pulls the pre-built image from the private **AWS ECR** repository (requires `docker login` to ECR — see below)
- Loads secrets (e.g. `DAGSHUB_PAT`) from a local `.env` file via `env_file`
- Publishes the API on `http://localhost:8000`
- Bind-mounts `./audit` on the host to `/pmg-api/audit` in the container, so the prediction **audit log persists outside the container**

### Build the image locally instead

If you want to build from source rather than pull from ECR (e.g. for local development), use the `Dockerfile` directly:

```bash
docker build -t mailguard .
docker run -d \
  --name mailguard \
  -p 8000:8000 \
  --env-file .env \
  -v $(pwd)/audit:/pmg-api/audit \
  mailguard
```

### What the image does

- Base image: `python:3.13-slim-bookworm`
- Installs `uv`, then runs `uv sync --frozen --no-dev` for a fast, locked, production-only install (dev dependencies like `ruff`, `matplotlib`, `ipykernel` are excluded)
- Copies in only what's needed to serve the model: `app/` and `artifacts/` (training code, data, and notebooks are **not** shipped in the image)
- Exposes port `8000` and starts the API with `uv run uvicorn app.app:app --host 0.0.0.0 --port 8000`

---

## ☁️ Production Deployment (AWS EC2 + ECR)

The live deployment (see [Live Demo](#-live-demo)) follows this workflow:

1. **Build & push to ECR**
   ```bash
   docker build -t mailguard .
   docker tag mailguard:latest 707578706440.dkr.ecr.us-east-1.amazonaws.com/sharanch33/mailguard:latest

   aws ecr get-login-password --region us-east-1 \
     | docker login --username AWS --password-stdin 707578706440.dkr.ecr.us-east-1.amazonaws.com

   docker push 707578706440.dkr.ecr.us-east-1.amazonaws.com/sharanch33/mailguard:latest
   ```

2. **Stage deployment config in S3** — `compose.yaml` and `.env` are uploaded to an S3 bucket, decoupling deployment configuration from the EC2 instance itself.

3. **Provision & configure the EC2 instance** — Docker (and Docker Compose) installed on the instance, and the instance's IAM role/credentials granted `ecr:GetAuthorizationToken` + pull access to the ECR repository.

4. **Pull config and run on EC2**
   ```bash
   # On the EC2 instance, inside the working directory
   aws s3 cp s3://<bucket-name>/compose.yaml .
   aws s3 cp s3://<bucket-name>/.env .

   aws ecr get-login-password --region us-east-1 \
     | docker login --username AWS --password-stdin 707578706440.dkr.ecr.us-east-1.amazonaws.com

   docker compose up -d
   ```

5. **Result** — the container runs on the instance with port `8000` exposed to the internet (via the instance's security group), and `./audit` bind-mounted so `audit/emails.csv` persists on the EC2 host across container restarts, redeploys, and image updates.

**Redeploying a new version** is just: rebuild → push to ECR with the same tag → on the EC2 host run `docker compose pull && docker compose up -d` to pull the new image and recreate the container.

> **Note on the audit log across deployments:** since `audit/` is bind-mounted to the EC2 host's filesystem (not baked into the image), the audit trail survives image updates and container restarts — but it lives only on that one instance's disk. It is **not** currently backed up to S3 or any other durable store, so a lost or terminated instance means a lost audit log. This would be a natural next improvement (see [Roadmap](#-roadmap)).

---

## 📡 API Reference

### `GET /`
Returns basic API metadata.

```json
{
  "message": "Spam Email Classifier API",
  "version": "1.0.0",
  "endpoints": {
    "health": "/health",
    "predict": "/predict",
    "docs": "/docs"
  }
}
```

### `GET /health`
Health check — confirms the model, vectorizer, and NLP pipeline are loaded.

```json
{
  "status": "healthy",
  "model_loaded": true,
  "vectorizer_loaded": true,
  "nlp_loaded": true
}
```

### `POST /predict`
Classifies a single email as **Spam** or **Not Spam**.

**Request body:**
```json
{
  "text": "Congratulations! You've won a $1000 gift card, click here to claim now!!!"
}
```

**Response:**
```json
{
  "prediction": "Spam",
  "confidence": 94.32,
  "probabilities": {
    "ham": 0.0568,
    "spam": 0.9432
  },
  "cleaned_text": "congratulation win gift card click claim"
}
```

| Field | Type | Description |
|---|---|---|
| `text` | `string` (min length 3) | Raw email text to classify |

### `GET /model/info`
Returns metadata about the currently loaded model and vectorizer.

```json
{
  "model_type": "Logistic Regression",
  "vectorizer_type": "TF-IDF",
  "max_features": 2000,
  "vocabulary_size": 2000,
  "ngram_range": [1, 2],
  "preprocessing": [
    "Lowercase conversion",
    "URL removal",
    "Email address removal",
    "HTML tag removal",
    "Special character removal",
    "Stopwords removal",
    "Lemmatization"
  ]
}
```

---

## 🤖 Model Details

- **Algorithm:** Logistic Regression (`scikit-learn`)
- **Feature Extraction:** TF-IDF, `max_features=2000`, `ngram_range=(1, 2)`
- **Hyperparameters** (`params.yaml`):
  - `C = 0.01`
  - `solver = liblinear`
  - `penalty = l2`
  - `random_state = 42`
- **Text Preprocessing Pipeline:**
  1. Lowercasing
  2. URL removal
  3. Email address removal
  4. HTML tag stripping
  5. Bracketed-text removal
  6. Special character / digit removal
  7. Whitespace normalization
  8. spaCy tokenization → stopword & punctuation removal → lemmatization
- **Evaluation Metrics** (`reports/metrics.json`): Accuracy, Precision, Recall, ROC-AUC

---

## ⚙️ Configuration

All tunable hyperparameters live in a single file, **`params.yaml`**, consumed by both the DVC pipeline and the training scripts:

```yaml
data-ingestion:
  test_size: 0.2

feature-engineering:
  max_features: 2000
  ngram_range: [1, 2]

model-building:
  C: 0.01
  solver: "liblinear"
  penalty: "l2"
  random_state: 42
```

Changing a value here and running `dvc repro` will automatically re-trigger only the affected downstream stages.

---

## 📊 Experiment Tracking

Every training run logs the following to **MLflow** (hosted via **DagsHub**):

- 📈 Metrics: `accuracy`, `precision`, `recall`, `auc`
- ⚙️ Parameters: `C`, `solver`, `penalty`, `random_state`
- 📦 Model artifact: registered under `MailGuard-API` in the Model Registry

The FastAPI service loads the model directly from the registry at startup via:

```python
mlflow.sklearn.load_model(model_uri="models:/MailGuard-API/1")
```

---

## 🧾 Request Auditing

Every successful call to `POST /predict` is appended as a new row to `audit/emails.csv` (created automatically on first request):

| Column | Description |
|---|---|
| `email` | The raw input text submitted for classification |
| `label` | The predicted label (`Spam` / `Not Spam`) |

This gives a simple, human-readable trail of everything the deployed model has classified — useful for spot-checking predictions, building a future retraining dataset, or debugging misclassifications in production.

- Locally, the file lives at `audit/emails.csv` relative to the project root.
- In Docker, `audit/` is bind-mounted to the host (see [Running with Docker](#-running-with-docker)), so the log survives container restarts and rebuilds.
- On the **live EC2 deployment**, this same bind mount means every request sent to [`http://54.83.143.31:8000/predict`](#-live-demo) is being logged in real time to `audit/emails.csv` on the instance — the live app is actively collecting real inference data, not just serving predictions.
- `audit/emails.csv` is git-ignored — only the empty `audit/` directory (via `audit/.gitignore`) is tracked in version control.

---

## 🪵 Logging

Each pipeline stage writes structured, timestamped logs to both the console and a dedicated file under `logs/`:

| Stage | Log File |
|---|---|
| Data Ingestion | `logs/ingestion.log` |
| Preprocessing | `logs/preprocessing.log` |
| Feature Engineering | `logs/feature_engineering.log` |
| Model Building | `logs/model_building.log` |
| Model Evaluation | `logs/model-evaluation.log` |

Format: `%(asctime)s - %(name)s - %(levelname)s - %(message)s`

---

## 🗺 Roadmap

- [x] Add a `Dockerfile` + container image for deployment
- [x] Add Docker Compose setup with persisted audit log volume
- [x] Add request auditing (`audit/emails.csv`)
- [x] Push the Docker image to a registry (**AWS ECR**) as part of the release flow
- [x] Deploy the container to a live host (**AWS EC2**)
- [ ] Automate the ECR build/push + EC2 deploy steps into a CI/CD workflow instead of running them manually
- [ ] Assign an Elastic IP (or put the instance behind a load balancer / DNS name) so the public URL doesn't change on instance restart
- [ ] Back up `audit/emails.csv` off the EC2 host (e.g. periodic sync to S3) so audit data survives instance loss
- [ ] Add HTTPS/TLS in front of the API (e.g. via a reverse proxy or load balancer) — it's currently served over plain HTTP
- [ ] Add batch prediction endpoint (`/predict/batch`)
- [ ] Add automated test suite (`pytest`)
- [ ] Add authentication/rate-limiting to the API

---

## 🤝 Contributing

Contributions, issues, and feature requests are welcome!

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

This project uses **Ruff** for linting/formatting — please run `ruff check .` and `ruff format .` before submitting a PR.

---

## 📄 License

No license file is currently published in this repository. All rights reserved by the author unless stated otherwise.

---

<div align="center">

Built by **[Sharanch3](https://github.com/Sharanch3)**

</div>
