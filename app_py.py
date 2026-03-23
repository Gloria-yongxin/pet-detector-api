"""
app.py  —  Oxford-IIIT Pet Head Detection Microservice
POST /predict  → JSON list of detected bounding boxes + confidence scores
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image
import io, os, urllib.request

# ── Config ───────────────────────────────────────────────────────────────────
MODEL_PATH  = "pet_detector.pth"
MODEL_URL   = os.getenv("MODEL_URL", "")   # set this in Render env vars
SCORE_THRESH = 0.5                          # minimum confidence to return a box
device = torch.device("cpu")

# ── Load model ───────────────────────────────────────────────────────────────
def load_model():
    if not os.path.exists(MODEL_PATH):
        if not MODEL_URL:
            raise RuntimeError("MODEL_URL env var not set and model file missing.")
        print("Downloading model weights …")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")

    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    m.roi_heads.box_predictor = FastRCNNPredictor(
        m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    return m

model = load_model()

# ── Image → tensor ───────────────────────────────────────────────────────────
to_tensor = T.ToTensor()

# ── FastAPI ──────────────────────────────────────────────────────────────────
app = FastAPI(
    title="Pet Head Detector",
    description=(
        "Upload a JPEG/PNG image of a cat or dog. "
        "The service returns bounding boxes around detected pet heads, "
        "each with a confidence score."
    ),
    version="1.0",
)

@app.get("/")
def root():
    return {
        "message": "Pet Head Detector is running.",
        "usage": "POST a JPEG/PNG image to /predict",
        "model": "Faster R-CNN ResNet-50 FPN fine-tuned on Oxford-IIIT Pet",
    }

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """
    Input : multipart/form-data with field 'file' containing a JPEG or PNG image.
    Output: JSON with a list of detections, each containing:
              - box        : [xmin, ymin, xmax, ymax]  (pixel coordinates)
              - confidence : float 0-1
              - label      : "pet_head"
    """
    # Validate content type
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400,
                            detail="Only JPEG/PNG images are accepted.")

    # Read image
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    img_w, img_h = img.size

    # Inference
    tensor = to_tensor(img).to(device)          # [3, H, W]
    with torch.no_grad():
        preds = model([tensor])[0]              # dict: boxes, labels, scores

    # Filter by confidence threshold
    keep   = preds["scores"] >= SCORE_THRESH
    boxes  = preds["boxes"][keep].tolist()
    scores = preds["scores"][keep].tolist()

    detections = [
        {
            "box":        [round(c, 1) for c in box],  # [xmin, ymin, xmax, ymax]
            "confidence": round(score, 4),
            "label":      "pet_head",
        }
        for box, score in zip(boxes, scores)
    ]

    return JSONResponse({
        "image_size":  {"width": img_w, "height": img_h},
        "num_detected": len(detections),
        "detections":  detections,
    })
