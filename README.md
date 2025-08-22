# Face Recognition Web App (InsightFace + Streamlit + SQLite)

A minimal, production-ready **prototype** for face recognition that:
- Detects faces on images (upload or camera).
- Identifies faces by comparing embeddings against a database.
- Outputs **name**, **gender**, **age (approximate)**.
- Supports **adding/registering** faces and **browsing** stored data.
- Pure Python backend with **SQLite** for persistence.

> Embedding & attributes powered by **InsightFace** (`buffalo_l`), which gives embeddings, age, and gender in one pass.

---

## 1) Environment Setup

```bash
python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r requirements.txt
```

> If you have a GPU and CUDA/cuDNN, you can change `CTX_ID` in `face_engine.py` to `0` to use GPU.

---

## 2) Run Locally

```bash
streamlit run app.py
```

Then open the local URL shown in the terminal.

---

## 3) How it Works (Quick Overview)

- `face_engine.py` loads `insightface.app.FaceAnalysis` with the `buffalo_l` model pack.
- For each detected face we get:
  - **embedding** (512-d, L2-normalized)
  - **age** (approx)
  - **gender** (`male`/`female`)
  - bounding box and a small **face crop** (for previews)
- Embeddings are stored in **SQLite** (table `faces`) as `float32` bytes; we keep a L2-normalized version so cosine similarity equals dot product.
- Matching is done by dot-product (cosine similarity) against all stored embeddings.
- If **similarity** is >= `SIM_THRESHOLD` (default **0.35**), we consider it a match; otherwise it is **Unknown**.

> You can tune `SIM_THRESHOLD` in `face_engine.py` (`SIM_THRESHOLD = 0.35`) to trade off precision/recall for your own data. Typical values are in the range `0.30–0.45` for ArcFace-like embeddings, but **please validate** on your own dataset.

---

## 4) Registering People

Use the **"Register Face"** tab to:
- Input a **name** (required) and optional **notes/info**.
- Upload one or multiple photos, or capture from camera.
- The app will save **all** detected faces in the photos to the person.

> Tip: register **several** images per person in different angles/lighting for better accuracy.

---

## 5) Recognition

Use the **"Recognize"** tab to:
- Upload a photo or capture with the camera.
- The app draws bounding boxes and shows: **Name • gender • age • similarity**.

If your database is empty, everything will be **Unknown** until you register some faces.

---

## 6) Database Tab

Browse people and see stored face crops. Data are persisted in `faces.db` (SQLite) in the app folder.

To reset, simply delete `faces.db`.

---

## 7) Deployment (Web)

### Option A — Streamlit Community Cloud (quickest)
1. Push this folder to a public GitHub repo.
2. On https://streamlit.io/cloud, create a new app from that repo.
3. Set the **Main file** to `app.py`. The default branch should work.
4. Set the **Python version** to `3.10` or later (Settings).
5. The app will install from `requirements.txt` and run automatically.

> Free tiers can be slow to cold-start the InsightFace models. First request might take longer.

### Option B — Your own server (Linux)
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py --server.address 0.0.0.0 --server.port 8501
```
Reverse-proxy with Nginx if needed.

### Option C — Docker (advanced; CPU example)
Create `Dockerfile`:
```dockerfile
FROM python:3.10-slim

# System deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential libgl1 libglib2.0-0 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.address", "0.0.0.0", "--server.port", "8501"]
```

Build & run:
```bash
docker build -t face-rec-app .
docker run --rm -p 8501:8501 face-rec-app
```

---

## 8) Notes & Tips

- **Threshold tuning**: export embeddings and evaluate on your own positive/negative pairs.
- **Data privacy**: this demo stores cropped faces in the DB. Remove `img` column if you must avoid storing images.
- **Scaling up**: move from SQLite to Postgres; use FAISS/Annoy for faster nearest-neighbor search with large datasets.
- **Security**: add authentication if exposing on the internet.
- **Legal**: ensure you have consent and comply with local regulations for biometric data.
