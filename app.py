from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import torch
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision import transforms as T
from PIL import Image
import io, os, urllib.request

MODEL_PATH = "pet_detector.pth"
MODEL_URL  = os.getenv("MODEL_URL", "")
device     = torch.device("cpu")

def load_model():
    if not os.path.exists(MODEL_PATH):
        if not MODEL_URL:
            raise RuntimeError("MODEL_URL env var not set and model file missing.")
        print("Downloading model weights...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
        print("Download complete.")
    m = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=None)
    m.roi_heads.box_predictor = FastRCNNPredictor(
        m.roi_heads.box_predictor.cls_score.in_features, 2)
    m.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    m.eval()
    return m

model     = load_model()
to_tensor = T.ToTensor()

app = FastAPI(
    title="Pet Head Detector",
    description="Upload a JPEG/PNG image of a cat or dog. Returns bounding boxes around detected pet heads.",
    version="1.0",
)

@app.get("/")
def root():
    return {"message": "Pet Head Detector is running. POST an image to /predict"}

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    if file.content_type not in ("image/jpeg", "image/png", "image/jpg"):
        raise HTTPException(status_code=400, detail="Only JPEG/PNG images are accepted.")
    contents = await file.read()
    try:
        img = Image.open(io.BytesIO(contents)).convert("RGB")
    except Exception:
        raise HTTPException(status_code=400, detail="Could not decode image.")

    img_w, img_h = img.size
    tensor = to_tensor(img).to(device)
    with torch.no_grad():
        preds = model([tensor])[0]

    keep       = preds["scores"] >= 0.5
    boxes      = preds["boxes"][keep].tolist()
    scores     = preds["scores"][keep].tolist()
    detections = [
        {"box": [round(c, 1) for c in box], "confidence": round(score, 4), "label": "pet_head"}
        for box, score in zip(boxes, scores)
    ]
    return JSONResponse({
        "image_size":   {"width": img_w, "height": img_h},
        "num_detected": len(detections),
        "detections":   detections,
    })
