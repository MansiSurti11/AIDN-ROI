import torch
from torchvision import transforms
from PIL import Image, ImageFilter
import numpy as np
import cv2
import io

def extract_patch(image: Image.Image, bbox: dict) -> Image.Image:
    """
    bbox = {"x": int, "y": int, "w": int, "h": int}
    Coordinates are in LR image pixel space.
    """
    x, y, w, h = bbox["x"], bbox["y"], bbox["w"], bbox["h"]
    # Ensure dimensions are divisible by some factor if required by model, 
    # but AIDN is scale-arbitrary.
    patch = image.crop((x, y, x + w, y + h))
    return patch

def restore_patch(patch: Image.Image, model, scale_factor: float, device='cpu') -> Image.Image:
    """
    Restore a single patch using the provided model.
    """
    # Convert patch to tensor [1, C, H, W]
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    tensor = transform(patch).unsqueeze(0).to(device)
    
    # Calculate target dimensions
    _, _, h, w = tensor.shape
    outH, outW = int(h * scale_factor), int(w * scale_factor)
    
    with torch.no_grad():
        # model expects (input, scale, outH, outW)
        restored_tensor = model(tensor, scale_factor, outH, outW)
    
    # Convert back to PIL
    restored = transforms.ToPILImage()(restored_tensor.squeeze(0).clamp(0, 1).cpu())
    return restored

def blend_patch_into_image(
    base_lr: Image.Image,
    hr_patch: Image.Image,
    bbox: dict,
    feather_px: int = 8,
    use_poisson: bool = True
) -> Image.Image:
    """
    Blend the HR patch back into the LR image.
    Since the HR patch is high-res, we first resize it down to the ROI size 
    to fit back into the LR image, OR we could upscale the whole image.
    The goal of this tool is specific ROI HR recovery, but the UI usually 
    shows a composite of the ROI in the original context.
    """
    lr_w, lr_h = bbox["w"], bbox["h"]
    
    # If we want a full HR image, we'd need to upscale the base_lr first.
    # But usually, we just want to see the "recovered" part in the original LR image.
    # We resize the HR patch to fit the LR bbox.
    hr_patch_resized = hr_patch.resize((lr_w, lr_h), Image.LANCZOS)
    
    if not use_poisson:
        # Alpha blending with feathering
        mask = Image.new("L", (lr_w, lr_h), 255)
        if feather_px > 0:
            # Create a black border for feathering
            inner_mask = Image.new("L", (lr_w - 2*feather_px, lr_h - 2*feather_px), 255)
            mask = Image.new("L", (lr_w, lr_h), 0)
            mask.paste(inner_mask, (feather_px, feather_px))
            mask = mask.filter(ImageFilter.GaussianBlur(feather_px))
        
        output = base_lr.copy()
        output.paste(hr_patch_resized, (bbox["x"], bbox["y"]), mask)
        return output
    else:
        # Poisson blending using OpenCV
        src = cv2.cvtColor(np.array(hr_patch_resized), cv2.COLOR_RGB2BGR)
        dst = cv2.cvtColor(np.array(base_lr), cv2.COLOR_RGB2BGR)
        
        # Binary mask
        mask = np.full(src.shape, 255, src.dtype)
        
        # Center of the ROI in the destination image
        center = (bbox["x"] + lr_w // 2, bbox["y"] + lr_h // 2)
        
        try:
            blended = cv2.seamlessClone(src, dst, mask, center, cv2.NORMAL_CLONE)
            return Image.fromarray(cv2.cvtColor(blended, cv2.COLOR_BGR2RGB))
        except Exception as e:
            print(f"Poisson blending failed: {e}. Falling back to alpha blending.")
            return blend_patch_into_image(base_lr, hr_patch, bbox, feather_px, use_poisson=False)

def create_comparison(lr_patch: Image.Image, hr_patch: Image.Image, scale_factor: float) -> Image.Image:
    """
    Create a side-by-side comparison.
    Resizes LR patch to match HR patch height.
    """
    hr_w, hr_h = hr_patch.size
    lr_patch_resized = lr_patch.resize((int(lr_patch.width * scale_factor), int(lr_patch.height * scale_factor)), Image.NEAREST)
    
    combined = Image.new("RGB", (lr_patch_resized.width + hr_w + 10, hr_h), (255, 255, 255))
    combined.paste(lr_patch_resized, (0, 0))
    combined.paste(hr_patch, (lr_patch_resized.width + 10, 0))
    return combined
