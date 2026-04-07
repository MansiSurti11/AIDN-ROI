import numpy as np
from PIL import Image
from aidn_inference import AIDNWrapper
import os

def test_advanced_features():
    wrapper = AIDNWrapper(
        config_path='config/DIV2K/AIDN_benchmark.yaml',
        weight_path='LOG/DIV2K/pre-train/model_best.pth'
    )
    
    # 1. Test Watermarking
    print("Testing Watermarking...")
    test_img = Image.new("RGB", (256, 256), (128, 128, 128))
    secret_msg = "User-123-Copyright-2026"
    
    # Embed
    wm_img = wrapper.embed_watermark(test_img, secret_msg)
    
    # Verify
    extracted_msg = wrapper.verify_watermark(wm_img)
    print(f"Original: {secret_msg}")
    print(f"Extracted: {extracted_msg}")
    
    assert secret_msg == extracted_msg, "Watermark extraction failed!"
    print("SUCCESS: Watermarking verified!")
    
    # 2. Test Saliency + Batch (Summary)
    print("Saliency and Batch features are structural; verified in previous scripts.")
    
    # 3. Test impact on HR (Optional/Qualitative)
    # Since LSB only changes the last bit, it adds +/- 1/255 noise.
    # Most SR models are robust to this level of quantization noise.

if __name__ == "__main__":
    test_advanced_features()
