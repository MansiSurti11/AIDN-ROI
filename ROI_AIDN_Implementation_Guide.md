# ROI-AIDN: Selective Region-of-Interest HR Recovery Tool

## Core Idea

This tool is built on top of **AIDN** (Scale-Arbitrary Invertible Image Downscaling Network, IEEE TIP 2023).

AIDN allows a sender to **embed full HD information invisibly inside a downscaled LR image** before uploading to social media. A receiver can later restore the original HR image from the LR version.

### Key property this tool exploits

From the original paper (Section IV-H.2): HR information is embedded **locally, not globally**. This means:
- A cropped patch from the embedded LR image can be restored independently
- You do NOT need to restore the full image to get HR detail for one region
- This enables selective, efficient, zoom-in style recovery

### What this tool adds

Instead of restoring the entire image (expensive, unnecessary), the user:
1. Uploads an embedded LR image
2. Draws a bounding box over a region of interest (ROI)
3. Gets back only that region restored to HR — stitched seamlessly back into the full image

---

## Tech Stack

| Layer | Choice |
|---|---|
| ML backend | Python, PyTorch |
| API | FastAPI |
| Frontend | React (Vite) or plain HTML/JS |
| Image processing | Pillow, OpenCV |
| AIDN model | Official pretrained weights from https://github.com/Doubiiu/AIDN |

---

## Project Structure

```
roi-aidn/
├── backend/
│   ├── main.py               # FastAPI app
│   ├── model.py              # AIDN model loader
│   ├── patch.py              # ROI crop, restore, blend logic
│   ├── requirements.txt
│   └── weights/
│       └── aidn.pth          # Downloaded pretrained weights
├── frontend/
│   ├── index.html
│   ├── app.js                # Upload + canvas bounding box UI
│   └── style.css
└── README.md
```

---

## Implementation Steps

### Step 1 — Set up AIDN model loader (`backend/model.py`)

- Clone the official AIDN repo: `git clone https://github.com/Doubiiu/AIDN`
- Copy the restoration network architecture code into `model.py`
- Download pretrained weights (link in the AIDN repo's README)
- Write a `load_restoration_network(weights_path)` function that:
  - Instantiates the restoration network (`Rφ`)
  - Loads weights with `torch.load`
  - Sets model to `eval()` mode
  - Returns the model

> Only the **restoration network** is needed here — the embedding network is the sender's concern.

---

### Step 2 — Patch extraction and restoration (`backend/patch.py`)

#### 2a. Crop the patch

```python
def extract_patch(image: PIL.Image, bbox: dict) -> PIL.Image:
    # bbox = {"x": int, "y": int, "w": int, "h": int}
    # coordinates are in LR image pixel space
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    patch = image.crop((x, y, x + w, y + h))
    return patch
```

#### 2b. Restore the patch

```python
def restore_patch(patch: PIL.Image, model, scale_factor: float) -> PIL.Image:
    # Convert patch to tensor
    tensor = transforms.ToTensor()(patch).unsqueeze(0)  # [1, C, H, W]
    
    # Pass scale factor — same value used during embedding
    with torch.no_grad():
        restored_tensor = model(tensor, scale_factor)
    
    # Convert back to PIL
    restored = transforms.ToPILImage()(restored_tensor.squeeze(0).clamp(0, 1))
    return restored
```

> The `scale_factor` must match what was used when the sender ran the embedding network. For a known platform (e.g. WhatsApp at 2048px), this can be computed automatically from original vs embedded dimensions.

#### 2c. Blend the restored patch back

```python
def blend_patch_into_image(
    base_lr: PIL.Image,
    hr_patch: PIL.Image,
    bbox: dict,
    feather_px: int = 8
) -> PIL.Image:
    # Resize hr_patch back to LR patch size for seamless composite
    lr_patch_size = (bbox["w"], bbox["h"])
    hr_patch_resized = hr_patch.resize(lr_patch_size, PIL.Image.LANCZOS)
    
    # Create output image as copy of LR
    output = base_lr.copy()
    
    # Paste with feathered alpha mask for smooth edges
    mask = create_feather_mask(lr_patch_size, feather_px)
    output.paste(hr_patch_resized, (bbox["x"], bbox["y"]), mask)
    
    return output

def create_feather_mask(size: tuple, feather_px: int) -> PIL.Image:
    # Create a white mask with soft edges using GaussianBlur
    mask = PIL.Image.new("L", size, 255)
    mask = ImageFilter.GaussianBlur(feather_px)(mask)
    return mask
```

> **Note on blending:** A simple feathered alpha paste is the baseline. If seams are visible, a more robust option is Poisson blending (OpenCV's `seamlessClone`).

---

### Step 3 — FastAPI backend (`backend/main.py`)

```python
from fastapi import FastAPI, UploadFile, Form
from fastapi.responses import StreamingResponse
import json

app = FastAPI()
model = load_restoration_network("weights/aidn.pth")

@app.post("/restore-roi")
async def restore_roi(
    image: UploadFile,
    bbox: str = Form(...),       # JSON string: {"x","y","w","h"}
    scale_factor: float = Form(...)
):
    bbox_dict = json.loads(bbox)
    
    # Load image
    pil_image = PIL.Image.open(image.file).convert("RGB")
    
    # Run patch pipeline
    patch = extract_patch(pil_image, bbox_dict)
    hr_patch = restore_patch(patch, model, scale_factor)
    composite = blend_patch_into_image(pil_image, hr_patch, bbox_dict)
    
    # Return composite as PNG
    buf = io.BytesIO()
    composite.save(buf, format="PNG")
    buf.seek(0)
    return StreamingResponse(buf, media_type="image/png")
```

Also add a `/compare-roi` endpoint that returns **both** the original LR patch and the restored HR patch side by side — useful for the frontend comparison view.

---

### Step 4 — Frontend UI (`frontend/app.js`)

#### Flow:
1. User uploads an embedded LR image → preview displayed on a `<canvas>`
2. User clicks and drags on the canvas to draw a bounding box
3. User inputs the scale factor (or it's auto-detected from image metadata if you store it)
4. On submit → POST to `/restore-roi` with image + bbox + scale_factor
5. Response image replaces the canvas content (composite shown)
6. Optional: show a split-view popup comparing LR patch vs HR patch

#### Key canvas logic:

```javascript
let startX, startY, bbox = null;

canvas.addEventListener("mousedown", (e) => {
  startX = e.offsetX;
  startY = e.offsetY;
});

canvas.addEventListener("mouseup", (e) => {
  bbox = {
    x: Math.min(startX, e.offsetX),
    y: Math.min(startY, e.offsetY),
    w: Math.abs(e.offsetX - startX),
    h: Math.abs(e.offsetY - startY)
  };
  drawBoundingBox(bbox);  // draw dashed rect on canvas overlay
});

async function submitROI() {
  const formData = new FormData();
  formData.append("image", imageFile);
  formData.append("bbox", JSON.stringify(bbox));
  formData.append("scale_factor", scaleFactor);

  const response = await fetch("/restore-roi", { method: "POST", body: formData });
  const blob = await response.blob();
  displayResult(URL.createObjectURL(blob));
}
```

---

### Step 5 — Multiple ROI support

- Maintain a list of bboxes: `roiList = []`
- After each restore, update the `base_lr` passed to the next call with the previous composite output
- This chains cleanly because each patch is locally independent

---

### Step 6 — Scale factor auto-detection (optional but recommended)

If you know the platform, compute scale factor from image dimensions:

```python
PLATFORM_MAX_DIM = {
    "whatsapp_hq": 2048,
    "whatsapp_data_saver": 1600,
    "wechat": 1080,
    "messenger": 1024,
    "facebook": 2048,
}

def compute_scale_factor(original_dim: int, platform: str) -> float:
    max_dim = PLATFORM_MAX_DIM[platform]
    return original_dim / max_dim
```

Or embed the scale factor as EXIF metadata in the LR image at embedding time so the receiver never has to input it manually.

---

## Key Implementation Notes

- **Do not run the embedding network** in this tool — only the restoration network is needed on the receiver side.
- The restoration network expects tensors in `[0, 1]` float range, not `[0, 255]` uint8.
- Match the `scale_factor` exactly to what was used during embedding — wrong scale = bad restoration.
- For the blending step, `cv2.seamlessClone` (Poisson blending) gives cleaner results than alpha feathering for high-contrast ROI boundaries.
- The paper trains AIDN with scale factors from ×1.1 to ×4.0 — stay within this range.
- AIDN processes features per-pixel so patch size doesn't need to match any fixed resolution — any crop works.

---

## Evaluation (for research paper writeup)

If using this as a research contribution, measure:

| Metric | What it measures |
|---|---|
| PSNR (patch) | Pixel fidelity of restored ROI vs ground truth |
| SSIM (patch) | Structural similarity of restored ROI |
| LPIPS (patch) | Perceptual quality of restored ROI |
| Composite PSNR | Quality of full composite (HR patch in LR context) |
| Inference time | Time for patch restore vs full image restore |
| Memory usage | Peak GPU memory for patch vs full image |

The compute savings claim (e.g. "X% faster than full-image restoration") is a key result — measure and report it.

---

## References

- AIDN paper: Xing et al., "Scale-Arbitrary Invertible Image Downscaling", IEEE TIP 2023
- AIDN GitHub: https://github.com/Doubiiu/AIDN
- FastAPI docs: https://fastapi.tiangolo.com
- OpenCV seamlessClone: https://docs.opencv.org/4.x/df/da0/group__photo__clone.html
