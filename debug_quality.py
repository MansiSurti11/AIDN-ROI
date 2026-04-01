import torch
import numpy as np
from PIL import Image
import torchvision.transforms.functional as TF
from aidn_inference import AIDNWrapper
import math

def calculate_psnr(img1, img2):
    i1 = np.array(img1).astype(np.float64)
    i2 = np.array(img2).astype(np.float64)
    mse = np.mean((i1 - i2) ** 2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(255.0 / math.sqrt(mse))

def main():
    # Load model
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    wrapper = AIDNWrapper(
        config_path='config/DIV2K/AIDN_benchmark.yaml',
        weight_path='LOG/DIV2K/pre-train/model_best.pth',
        device=device
    )

    # Load a test image (using the cat image we know is there)
    hr_path = 'assets/cat.png' 
    if not os.path.exists(hr_path):
        # Fallback if cat.png is missing, use any png in assets
        import glob
        pngs = glob.glob('assets/*.png')
        if pngs:
            hr_path = pngs[0]
        else:
            print("No test image found in assets/")
            return

    hr_img = Image.open(hr_path).convert("RGB")
    # Resize to have a nice patchable size
    hr_img = hr_img.resize((512, 512), Image.LANCZOS)
    
    print(f"Testing quality on: {hr_path}")
    print("-" * 30)

    for scale in [2.0, 4.0]:
        # 1. Embed
        lr_img = wrapper.embed(hr_img, scale)
        
        # 2. Restore full image (as a patch)
        bbox = (0, 0, lr_img.size[0], lr_img.size[1])
        restored_hr = wrapper.restore_patch(lr_img, bbox, scale)
        
        # 3. Compare with original HR (resized if needed)
        # We need the original HR to match the restored dimensions
        target_w, target_h = restored_hr.size
        orig_hr_matched = hr_img.crop((0, 0, target_w, target_h))
        
        psnr = calculate_psnr(orig_hr_matched, restored_hr)
        print(f"Scale {scale}x | PSNR: {psnr:.2f} dB")

if __name__ == "__main__":
    import os
    main()
