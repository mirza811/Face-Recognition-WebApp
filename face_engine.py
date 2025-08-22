\
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Optional, Dict
import numpy as np
from PIL import Image
import io
import cv2

# InsightFace
from insightface.app import FaceAnalysis

# ---------- Config ----------
MODEL_NAME = "buffalo_l"   # includes detection, recognition, age, gender
DET_SIZE = (640, 640)
CTX_ID = -1  # CPU; set 0 for GPU if available
SIM_THRESHOLD = 0.35  # cosine similarity threshold (unit vectors). Tune for your data.

# ---------- Utilities ----------
def l2_normalize(v: np.ndarray, axis: int = 1, eps: float = 1e-12) -> np.ndarray:
    norm = np.linalg.norm(v, axis=axis, keepdims=True)
    return v / np.maximum(norm, eps)

def pil_to_bgr(img: Image.Image) -> np.ndarray:
    if img.mode != "RGB":
        img = img.convert("RGB")
    arr = np.array(img)
    bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)
    return bgr

def crop_bbox_rgb(img_rgb: np.ndarray, bbox: np.ndarray, pad: int = 2) -> np.ndarray:
    h, w = img_rgb.shape[:2]
    x1, y1, x2, y2 = bbox.astype(int)
    x1 = max(x1 - pad, 0)
    y1 = max(y1 - pad, 0)
    x2 = min(x2 + pad, w)
    y2 = min(y2 + pad, h)
    return img_rgb[y1:y2, x1:x2]

def rgb_to_bytes(img_rgb: np.ndarray, format: str = "JPEG") -> bytes:
    pil_img = Image.fromarray(img_rgb)
    bio = io.BytesIO()
    pil_img.save(bio, format=format, quality=90)
    return bio.getvalue()

@dataclass
class FaceResult:
    bbox: np.ndarray
    kps: np.ndarray
    det_score: float
    embedding: np.ndarray  # L2-normalized
    age: Optional[int]
    gender: Optional[str]  # "male"/"female"
    crop_bytes: Optional[bytes]

class FaceEngine:
    def __init__(self):
        self.app = FaceAnalysis(name=MODEL_NAME)
        self.app.prepare(ctx_id=CTX_ID, det_size=DET_SIZE)

    def extract(self, pil_img: Image.Image) -> List[FaceResult]:
        bgr = pil_to_bgr(pil_img)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        faces = self.app.get(bgr)  # returns list of Face objects
        results: List[FaceResult] = []
        for f in faces:
            emb = f.normed_embedding if hasattr(f, "normed_embedding") else l2_normalize(f.embedding.reshape(1, -1))[0]
            bbox = f.bbox.astype(int)
            crop_rgb = crop_bbox_rgb(rgb, bbox)
            crop_bytes = rgb_to_bytes(crop_rgb, format="JPEG")
            age = int(getattr(f, "age", -1)) if getattr(f, "age", None) is not None else None
            sex = getattr(f, "sex", None)
            if sex is None:
                gender = None
            else:
                # InsightFace returns 'male'/'female' in recent versions; sometimes 0/1
                gender = sex if isinstance(sex, str) else ("male" if sex == 1 else "female")
            results.append(FaceResult(
                bbox=bbox,
                kps=f.kps,
                det_score=float(f.det_score),
                embedding=emb.astype(np.float32),
                age=age,
                gender=gender,
                crop_bytes=crop_bytes
            ))
        return results

    def match(self, query_emb: np.ndarray, db_embs: np.ndarray) -> Tuple[int, float]:
        """
        Return (index, similarity) of best match, or (-1, 0.0) if nothing above threshold.
        query_emb: shape (D,), L2-normalized
        db_embs: shape (N, D), L2-normalized
        """
        if db_embs.size == 0:
            return -1, 0.0
        sims = db_embs @ query_emb  # cosine similarity since both are unit vectors
        idx = int(np.argmax(sims))
        sim = float(sims[idx])
        if sim >= SIM_THRESHOLD:
            return idx, sim
        return -1, sim
