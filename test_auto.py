import torch
import torchvision
from PIL import Image
import os
import time
from aidn_inference import AIDNWrapper

img = Image.new('RGB', (480, 240), color=(200, 150, 100))

print("=== AIDN Automatic Test ===")
print("System info:")
print(f"PyTorch version: {torch.__version__}")
print(f"Torchvision version: {torchvision.__version__}")
print("CUDA Available: ", torch.cuda.is_available())

start_load = time.time()
print("\nLoading AIDNWrapper...")
wrapper = AIDNWrapper(
    config_path='config/DIV2K/AIDN_benchmark.yaml',
    weight_path='LOG/DIV2K/pre-train/model_best.pth'
)
print(f"Loaded successfully in {time.time()-start_load:.2f} seconds.")

scale = 2.0
print(f"\n--- Embedding Test ---")
print(f"Input HR Image Size: {img.size}")
start_embed = time.time()
embedded_lr = wrapper.embed(img, scale)
print(f"Embed pass complete in {time.time()-start_embed:.2f} seconds.")
print(f"Output Embedded LR Image Size: {embedded_lr.size}")

print(f"\n--- Patch Restoration Test ---")
bbox_lr = (64, 64, 64, 64)
print(f"Selecting LR Patch BBox: {bbox_lr}")
start_restore = time.time()
restored_patch = wrapper.restore_patch(embedded_lr, bbox_lr, scale)
print(f"Restore pass complete in {time.time()-start_restore:.2f} seconds.")
print(f"Restored HR Patch Size: {restored_patch.size}")
print("\nTest completed successfully!")
