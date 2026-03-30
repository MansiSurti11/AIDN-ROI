import torch
import math
from torchvision import transforms
from PIL import Image
import numpy as np
import io

def get_safe_dimensions(image: Image.Image, max_dim=2560) -> Image.Image:
    """
    Resize image if it exceeds max_dim to prevent memory allocation errors.
    """
    w, h = image.size
    if max(w, h) > max_dim:
        scale = max_dim / max(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        # Ensure dimensions are even (AIDN preferred)
        new_w = new_w - (new_w % 2)
        new_h = new_h - (new_h % 2)
        print(f"Safe Scale: Resizing {w}x{h} to {new_w}x{new_h}")
        return image.resize((new_w, new_h), Image.LANCZOS)
    return image

def encode_image(hr_image: Image.Image, model, scale_factor: float, device='cpu', progress_callback=None) -> Image.Image:
    """
    Encode an HR image into an AIDN-embedded LR image using tiled processing.
    """
    # Safe resize to prevent memory issues
    hr_image = get_safe_dimensions(hr_image)
    
    w, h = hr_image.size
    tile_size = 512
    overlap = 32
    
    # Calculate grids
    w_num = math.ceil((w - overlap) / (tile_size - overlap))
    h_num = math.ceil((h - overlap) / (tile_size - overlap))
    total_tiles = w_num * h_num
    
    # Empty canvas for result
    lr_w, lr_h = int(w / scale_factor), int(h / scale_factor)
    lr_result = Image.new("RGB", (lr_w, lr_h))
    
    transform = transforms.ToTensor()
    
    for i in range(h_num):
        for j in range(w_num):
            # Calculate input tile coordinates (HR space)
            x0 = j * (tile_size - overlap)
            y0 = i * (tile_size - overlap)
            x1 = min(x0 + tile_size, w)
            y1 = min(y0 + tile_size, h)
            
            tile = hr_image.crop((x0, y0, x1, y1))
            tensor = transform(tile).unsqueeze(0).to(device)
            
            with torch.no_grad():
                lr_tile_tensor = model(tensor, 1.0 / scale_factor)
            
            lr_tile = transforms.ToPILImage()(lr_tile_tensor.squeeze(0).clamp(0, 1).cpu())
            
            # Calculate output tile coordinates (LR space)
            # Simple division since AIDN is pixel-to-pixel
            lx0, ly0 = int(x0 / scale_factor), int(y0 / scale_factor)
            
            # Paste into global canvas
            lr_result.paste(lr_tile, (lx0, ly0))
            
            # Progress callback
            if progress_callback:
                current_tile = i * w_num + j + 1
                progress_callback(int((current_tile / total_tiles) * 100))
                
    return lr_result
