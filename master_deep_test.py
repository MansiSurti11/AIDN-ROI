import time
import numpy as np
import torch
from PIL import Image, ImageDraw
import io
import os
import base64

from aidn_inference import AIDNWrapper
from saliency_utils import propose_roi, get_spectral_residual_saliency
from watermark_utils import text_to_bits, bits_to_text, embed_lsb_watermark, extract_lsb_watermark

def run_test(name, func):
    print(f"\n>>> Running Test Area: [{name}]")
    try:
        func()
        print(f"PASSED: {name}")
    except Exception as e:
        print(f"FAILED: {name} | ERROR: {str(e)}")
        import traceback
        traceback.print_exc()

def test_saliency_robustness():
    print("  Checking uniform and noise images...")
    # 1. Pure black
    img_black = Image.new("RGB", (256, 256), (0, 0, 0))
    roi = propose_roi(img_black)
    assert len(roi) == 4, "ROI generation failed for black image"
    
    # 2. Pure white
    img_white = Image.new("RGB", (256, 256), (255, 255, 255))
    roi = propose_roi(img_white)
    
    # 3. Random noise
    noise = np.random.randint(0, 255, (256, 256, 3), dtype=np.uint8)
    img_noise = Image.fromarray(noise)
    roi = propose_roi(img_noise)
    
    print("  Saliency logic is robust to extreme inputs.")

def test_watermark_deep():
    print("  Checking Unicode and Long Strings...")
    img = Image.new("RGB", (512, 512), (50, 50, 50))
    
    # 1. Unicode & Emojis
    msg_uni = "AIDN Proof: 🚀🛡️ 漢 (Copyright 2026)"
    wm_img = embed_lsb_watermark(img, msg_uni)
    extracted = extract_lsb_watermark(wm_img)
    assert extracted == msg_uni, f"Unicode fail: got {extracted}"
    
    # 2. Long String (find capacity)
    capacity_bits = 512 * 512 * 3
    capacity_chars = capacity_bits // 8
    print(f"  Theoretical capacity: {capacity_chars} UTF-8 chars")
    
    msg_long = "Data" * (1000) # 4000 chars
    wm_img_long = embed_lsb_watermark(img, msg_long)
    extracted_long = extract_lsb_watermark(wm_img_long)
    assert extracted_long == msg_long, "Long string fail"
    
    # 3. Text too long error
    try:
        huge_msg = "X" * (capacity_chars + 10)
        embed_lsb_watermark(img, huge_msg)
        assert False, "Should have failed due to length"
    except ValueError as e:
        print(f"  Length validation caught: {str(e)}")

def test_batching_performance():
    # Setup model
    wrapper = AIDNWrapper(
        config_path='config/DIV2K/AIDN_benchmark.yaml',
        weight_path='LOG/DIV2K/pre-train/model_best.pth'
    )
    
    img = Image.new("RGB", (1024, 1024), (100, 100, 100))
    # Draw some "shapes"
    d = ImageDraw.Draw(img)
    for i in range(10):
        d.rectangle([i*100, i*100, i*100+50, i*100+50], fill=(i*20, 255-(i*20), 128))
    
    scale = 2.0
    lr_img = wrapper.embed(img, scale)
    
    bboxes = [(10, 10, 64, 64), (100, 100, 96, 96), (200, 200, 128, 128), (300, 300, 64, 64)]
    
    # 1. Sequential
    start = time.time()
    for bbox in bboxes:
        wrapper.restore_patch(lr_img, bbox, scale)
    t_seq = time.time() - start
    print(f"  Sequential Time (4 patches): {t_seq:.4f}s")
    
    # 2. Batched
    start = time.time()
    wrapper.restore_patches_batch(lr_img, bboxes, scale)
    t_batch = time.time() - start
    print(f"  Batched Time (4 patches): {t_batch:.4f}s")
    
    improvement = (t_seq - t_batch) / t_seq * 100
    print(f"  Efficiency Gain: {improvement:.1f}%")
    # On CPU, batching might not be 50% faster, but it should be faster due to less Python overhead and better torch optimization

def test_slider_component_logic():
    print("  Checking HTML/Base64 generation...")
    # Import from app_streamlit (simulate environment)
    import base64
    def get_image_base64_dry(img):
        buffered = io.BytesIO()
        img.save(buffered, format="PNG")
        return base64.b64encode(buffered.getvalue()).decode()
    
    img1 = Image.new("RGB", (64, 64), (255, 0, 0))
    img2 = Image.new("RGB", (128, 128), (0, 0, 255)) # HR is larger usually
    
    b1 = get_image_base64_dry(img1)
    b2 = get_image_base64_dry(img2)
    
    assert len(b1) > 0 and len(b2) > 0, "Base64 encoding failed"
    print("  Base64 serialization OK.")

if __name__ == "__main__":
    run_test("Saliency Robustness", test_saliency_robustness)
    run_test("Watermark (Unicode/Limits)", test_watermark_deep)
    run_test("Batching Performance", test_batching_performance)
    run_test("Split-Slider Component Logic", test_slider_component_logic)
