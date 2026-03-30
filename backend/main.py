from fastapi import FastAPI, UploadFile, Form, File, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import json
import io
import PIL.Image as Image
import torch
import os

from .model import load_restoration_network, load_encoder_network
from .patch import extract_patch, restore_patch, blend_patch_into_image, create_comparison
from .embed import encode_image

app = FastAPI(title="ROI-AIDN API")

# Enable CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model instance
WEIGHTS_PATH = "backend/weights/aidn.pth"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
model = None
encoder_model = None

# Global progress store for encoding tasks
progress_store = {}

@app.on_event("startup")
async def startup_event():
    global model, encoder_model
    if os.path.exists(WEIGHTS_PATH):
        model = load_restoration_network(WEIGHTS_PATH, device=device)
        encoder_model = load_encoder_network(WEIGHTS_PATH, device=device)
        print(f"Models loaded successfully on {device}")
    else:
        print(f"Warning: Weights not found at {WEIGHTS_PATH}. Models will be initialized randomly.")
        model = load_restoration_network(None, device=device)
        encoder_model = load_encoder_network(None, device=device)

@app.post("/restore-roi")
async def restore_roi(
    image: UploadFile = File(...),
    bbox: str = Form(...),       # JSON string: {"x": int, "y": int, "w": int, "h": int}
    scale_factor: float = Form(...)
):
    try:
        bbox_dict = json.loads(bbox)
        
        # Load image
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Run patch pipeline
        patch = extract_patch(pil_image, bbox_dict)
        hr_patch = restore_patch(patch, model, scale_factor, device=device)
        
        # Blend ROI back into LR image
        composite = blend_patch_into_image(pil_image, hr_patch, bbox_dict, use_poisson=True)
        
        # Return composite as PNG
        buf = io.BytesIO()
        composite.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/compare-roi")
async def compare_roi(
    image: UploadFile = File(...),
    bbox: str = Form(...),       # JSON string: {"x": int, "y": int, "w": int, "h": int}
    scale_factor: float = Form(...)
):
    try:
        bbox_dict = json.loads(bbox)
        
        # Load image
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Run patch pipeline
        patch = extract_patch(pil_image, bbox_dict)
        hr_patch = restore_patch(patch, model, scale_factor, device=device)
        
        # Create side-by-side comparison
        combined = create_comparison(patch, hr_patch, scale_factor)
        
        # Return composite as PNG
        buf = io.BytesIO()
        combined.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.post("/embed")
async def embed(
    image: UploadFile = File(...),
    scale_factor: float = Form(...),
    task_id: str = Form(None)
):
    try:
        if task_id:
            progress_store[task_id] = 0
            
        # Load image
        img_bytes = await image.read()
        pil_image = Image.open(io.BytesIO(img_bytes)).convert("RGB")
        
        # Log resolution
        w, h = pil_image.size
        print(f"Received HR image for encoding: {w}x{h} (Task: {task_id})")
        
        # Define progress callback
        def update_progress(percent):
            if task_id:
                progress_store[task_id] = percent
        
        # Run encoder with tiling
        lr_image = encode_image(pil_image, encoder_model, scale_factor, device=device, progress_callback=update_progress)
        
        # Finalize progress
        if task_id:
            progress_store[task_id] = 100
            
        # Return LR image as PNG
        buf = io.BytesIO()
        lr_image.save(buf, format="PNG")
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/png")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": str(e)})

@app.get("/encoding-progress/{task_id}")
async def get_encoding_progress(task_id: str):
    progress = progress_store.get(task_id, 0)
    return {"progress": progress}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
