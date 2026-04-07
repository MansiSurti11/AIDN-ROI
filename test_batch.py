import torch
import numpy as np
from PIL import Image
from aidn_inference import AIDNWrapper
import os, time

def test_batch():
    # 1. Setup Wrapper
    wrapper = AIDNWrapper(
        config_path='config/DIV2K/AIDN_benchmark.yaml',
        weight_path='LOG/DIV2K/pre-train/model_best.pth'
    )
    
    # 2. Create/Load image
    if os.path.exists("hd.jpg"):
        hr_img = Image.open("hd.jpg").convert("RGB")
    else:
        # Create dummy if hd.jpg not found
        hr_img = Image.new("RGB", (800, 800), (100, 100, 100))
    
    scale = 2.0
    print(f"Embedding image... (scale {scale})")
    lr_img = wrapper.embed(hr_img, scale)
    
    # 3. Define multiple ROIs of different sizes
    bboxes = [
        (10, 10, 64, 64),
        (100, 100, 128, 96),
        (200, 50, 48, 120),
        (300, 300, 200, 200)
    ]
    
    print(f"Starting batched restoration of {len(bboxes)} regions...")
    start_time = time.time()
    results = wrapper.restore_patches_batch(lr_img, bboxes, scale)
    end_time = time.time()
    
    print(f"Batched inference took {end_time - start_time:.4f} seconds.")
    
    # 4. Verify results
    assert len(results) == len(bboxes), "Result count mismatch!"
    
    for i, (res, bbox) in enumerate(zip(results, bboxes)):
        expected_w = round(bbox[2] * scale)
        expected_h = round(bbox[3] * scale)
        print(f"ROI #{i+1}: expected {expected_w}x{expected_h}, got {res.size[0]}x{res.size[1]}")
        assert res.size == (expected_w, expected_h), f"Size mismatch for ROI #{i+1}"
        
    print("SUCCESS: Batched restoration verified!")

if __name__ == "__main__":
    test_batch()
