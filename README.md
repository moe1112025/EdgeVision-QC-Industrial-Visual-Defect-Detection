# EdgeVision QC — Industrial Visual Defect Detection

EdgeVision QC is a standalone computer-vision reference system using PyTorch for image classification, FastAPI for inference, and Streamlit for the inspection interface.

## Major Upgrade From the Legacy Prototype

The original implementation created a CNN and immediately called `eval()` without loading trained weights. This edition contains a reproducible training pipeline, checkpoint metadata, validation, adaptive pooling and a proper API boundary.

## Capabilities

- Folder-based labeled image dataset
- Synthetic demo dataset generator
- Train/validation split
- PyTorch CNN with BatchNorm and AdaptiveAvgPool2d
- Checkpoint serialization
- FastAPI health and prediction endpoints
- Streamlit inspection dashboard
- Input validation and HTTP error handling
- Unit tests for the model shape

## Demo Dataset

Generate local demo imagery:

```bash
python scripts/create_demo_dataset.py
```

The generated data is for software validation only and is not a production inspection benchmark.

## Training

```bash
python train.py
```

## Start API

```bash
uvicorn main:app --host 127.0.0.1 --port 8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## Start Dashboard

```bash
streamlit run app.py
```

## API Contract

`GET /health`

`POST /predict/` with multipart field `file`.

Example response:

```json
{
  "status": "success",
  "prediction": "Normal / Pass",
  "confidence": 97.12,
  "filename": "part_001.png"
}
```

## Technology

PyTorch: https://pytorch.org/
FastAPI: https://fastapi.tiangolo.com/
OpenCV: https://docs.opencv.org/
Streamlit: https://docs.streamlit.io/

## License

Project source: MIT. Production deployment requires representative labeled data, validation and domain-specific quality controls.
