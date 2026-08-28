# Cats vs Dogs MLOps Pipeline

MLOps ((S1-25_AIMLCZG523)) — Assignment 2
BITS Pilani WILP
Submitted by: Akash Kumar (2024AC05843)

**Repository:** https://github.com/akashkt09/cats-dogs-mlops
**Video walkthrough:** https://drive.google.com/file/d/1JKue9VL1rzvDEXmIhiagS8SNA1g2bnmD/view?usp=share_link

An end-to-end MLOps pipeline for a binary image classifier (cats vs dogs) built
for a pet adoption platform use case — covering model development, experiment
tracking, containerization, CI/CD, deployment on Kubernetes, and monitoring.

---

## Project Structure

```
cats-dogs-mlops/
├── app/
│   ├── main.py              # FastAPI inference service
│   ├── Dockerfile
│   └── requirements.txt
├── data/
│   ├── training_set.dvc     # DVC pointer — raw images versioned via Google Drive
│   └── test_set.dvc
├── deployment/
│   └── k8s-manifest.yaml    # Kubernetes Deployment + Service
├── model/
│   ├── cats_dogs_cnn.h5     # Trained CNN
│   └── cats_dogs_cnn.h5.dvc
├── notebooks/
│   └── cats_dogs_training.ipynb   # EDA, training, MLflow logging (run on Colab)
├── scripts/
│   └── post_deployment_check.py   # Sends labeled test images to the live API
├── screenshots/
├── tests/
│   ├── sample_images/sample_cat.jpg   # Committed fixture for CI smoke tests
│   ├── test_preprocessing.py
│   └── test_inference.py
├── .github/workflows/
│   ├── ci.yml                # Test → build → push to GitHub Container Registry
│   └── cd.yml                # Deploy to local Kubernetes via self-hosted runner
├── requirements.txt
└── README.md
```

---

## Setup / Install Instructions

### 1. Clone and set up Python

```bash
git clone https://github.com/akashkt09/cats-dogs-mlops.git
cd cats-dogs-mlops
pyenv local 3.11.9
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Python 3.11.9 is required — TensorFlow does not currently publish wheels for
newer versions (3.13+), which will fail to install otherwise.

### 2. Pull the dataset and model via DVC

The raw dataset (~10,000 images) and the trained model are versioned with DVC
and stored on Google Drive, not committed directly to Git.

```bash
pip install "dvc[gdrive]"
pip install --upgrade pyopenssl cryptography   # avoids a known gdrive auth error
dvc pull
```

This will prompt a one-time Google OAuth flow in your browser on first run.

### 3. Run the training pipeline (optional — a trained model is already included)

The full EDA, training, and MLflow logging workflow is in
`notebooks/cats_dogs_training.ipynb`, designed to run on Google Colab with GPU
access. It downloads the dataset from Kaggle, trains a CNN, evaluates it, and
logs the run to MLflow.

### 4. Run the inference API locally

```bash
cd app
pip install -r requirements.txt
python3 -m uvicorn main:app --host 0.0.0.0 --port 8000
```

Visit `http://localhost:8000/docs` for the interactive Swagger UI, or:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -F "file=@../tests/sample_images/sample_cat.jpg"
```

### 5. Run the unit tests

```bash
python3 -m pytest tests/ -v
```

### 6. Build and run with Docker

```bash
cd app
docker build -t cats-dogs-api:v1 .
docker run -d -p 8000:8000 cats-dogs-api:v1
curl http://localhost:8000/health
```

---

## CI/CD Pipeline

**CI** (`.github/workflows/ci.yml`) runs on every push to `main`:
1. Checks out the repo and installs dependencies
2. Runs the unit test suite (pytest)
3. Pulls the trained model from the DVC remote (Google Drive)
4. Builds the Docker image
5. Pushes it to GitHub Container Registry (`ghcr.io`), tagged both `latest` and
   with the commit SHA

**CD** (`.github/workflows/cd.yml`) runs after CI succeeds, on a **self-hosted
runner** (since the deployment target is a local Kubernetes cluster that a
GitHub-hosted cloud runner cannot reach):
1. Applies the Kubernetes manifest and restarts the deployment, pulling the
   newly built image
2. Waits for the rollout to complete
3. Runs two smoke tests — a health check and a real prediction call — and
   fails the pipeline if either fails

To run the self-hosted runner locally:

```bash
cd actions-runner
./run.sh
```

This must be running for CD to execute; it listens for jobs dispatched from
GitHub Actions.

---

## Deployed API — Access Instructions (Local Testing)

No public URL — deployed and tested locally via Kubernetes on Docker Desktop.

### 1. Prerequisites

- Docker Desktop with Kubernetes enabled
- `kubectl` CLI

### 2. Deploy

```bash
kubectl apply -f deployment/k8s-manifest.yaml
kubectl get pods
kubectl get services
```

Both pods should show `1/1 Running`, and `cats-dogs-api-service` should be a
`LoadBalancer` on port `8080`.

### 3. Test

```bash
curl http://localhost:8080/health

curl -X POST http://localhost:8080/predict \
  -F "file=@tests/sample_images/sample_cat.jpg"

curl http://localhost:8080/metrics
```

### 4. Post-deployment performance check

Sends a batch of 30 real, labeled test images to the live deployed API and
reports accuracy:

```bash
python3 scripts/post_deployment_check.py
```

---

## Model Summary

- **Architecture:** Simple CNN (3 conv blocks, 32→64→128 filters, dropout 0.5)
- **Input:** 224×224 RGB images, augmented at train time (flip, rotation, zoom)
- **Training:** 10 epochs, Adam optimizer, binary cross-entropy loss
- **Test accuracy:** 77.7% | **Test ROC-AUC:** 0.884
- **Post-deployment accuracy** (30 live requests against the deployed API): 76.7%,
  consistent with offline test performance — dogs are classified more
  reliably than cats (a real, observed asymmetry, not a deployment artifact)

---

## Monitoring

The inference service logs every prediction request (filename, content type,
prediction, confidence, response time — never the raw image bytes) to both a
log file and stdout, and exposes `/metrics` with request counts, error counts,
prediction distribution, and average latency.

**Known limitation:** metrics are held in-memory per pod. With 2 replicas
behind a load-balanced Service, a `/metrics` call can land on a different pod
than the one that served a recent prediction, so the counters can appear
inconsistent depending on routing. A production fix would use a centralized
store (e.g., Prometheus scraping all pods).

---

## Deliverables

1. **Submission zip** — full source code, DVC config/pointer files, CI/CD
   workflows, Docker/Kubernetes configs, and the trained model artifact
2. **Video walkthrough** (<5 min) — linked at the top of this document,
   demonstrating a code change through to a live deployed prediction
