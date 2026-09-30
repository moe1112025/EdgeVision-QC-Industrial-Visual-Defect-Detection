from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F
from fastapi import FastAPI, File, HTTPException, UploadFile

from model import DefectDetectorCNN

BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR / "artifacts" / "defect_detector.pt"
CLASSES = ["Normal / Pass", "Defective / Fail"]
IMAGE_SIZE = 128
app = FastAPI(title="EdgeVision QC API", version="1.0.0")
_model = None


def get_model():
    global _model
    if _model is None:
        if not MODEL_PATH.exists():
            raise FileNotFoundError("Run python train.py before inference")
        checkpoint = torch.load(MODEL_PATH, map_location="cpu", weights_only=False)
        _model = DefectDetectorCNN(num_classes=len(CLASSES))
        _model.load_state_dict(checkpoint["state_dict"])
        _model.eval()
    return _model


@app.get("/")
def root():
    return {"service": "EdgeVision QC API", "status": "running", "model_available": MODEL_PATH.exists()}


@app.get("/health")
def health():
    return {"status": "ok", "model_available": MODEL_PATH.exists()}


@app.post("/predict/")
async def predict(file: UploadFile = File(...)):
    if not file.content_type or not file.content_type.startswith("image/"):
        raise HTTPException(status_code=415, detail="Only image uploads are supported")
    try:
        raw = await file.read()
        array = np.frombuffer(raw, dtype=np.uint8)
        frame = cv2.imdecode(array, cv2.IMREAD_COLOR)
        if frame is None:
            raise HTTPException(status_code=400, detail="Invalid image")
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame = cv2.resize(frame, (IMAGE_SIZE, IMAGE_SIZE), interpolation=cv2.INTER_AREA)
        tensor = torch.from_numpy(frame.astype(np.float32) / 255.0).permute(2, 0, 1).unsqueeze(0)
        tensor = (tensor - 0.5) / 0.5
        with torch.no_grad():
            probabilities = F.softmax(get_model()(tensor), dim=1)[0]
        confidence, index = torch.max(probabilities, dim=0)
        return {"status": "success", "prediction": CLASSES[int(index)], "confidence": round(float(confidence) * 100, 2), "filename": file.filename}
    except HTTPException:
        raise
    except FileNotFoundError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Inference failed: {exc}") from exc
