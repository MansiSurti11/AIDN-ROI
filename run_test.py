import os
import torch
from PIL import Image
from aidn_inference import AIDNWrapper

def main():
    # Setup
    output_dir = 'output'
    os.makedirs(output_dir, exist_ok=True)
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    wrapper = AIDNWrapper(
        config_path='config/DIV2K/AIDN_benchmark.yaml',
        weight_path='LOG/DIV2K/pre-train/model_best.pth',
        device=device
    )

    # 1. Load hd.jpg
    input_path = 'hd.jpg'
    if not os.path.exists(input_path):
        print(f"Error: {input_path} not found!")
        return

    print(f"Processing {input_path}...")
    hr_img = Image.open(input_path).convert("RGB")
    
    # 2. Embed (LR) at 2x scale
    scale = 2.0
    with torch.no_grad():
        lr_img = wrapper.embed(hr_img, scale)
    
    lr_path = os.path.join(output_dir, 'hd_lr.png')
    lr_img.save(lr_path)
    print(f"Saved LR image to: {lr_path} ({lr_img.size[0]}x{lr_img.size[1]})")

    # 3. Restore (HR) using tiled restoration
    with torch.no_grad():
        restored_hr = wrapper.restore_full_image(lr_img, scale)
    
    hr_path = os.path.join(output_dir, 'hd_restored.png')
    restored_hr.save(hr_path)
    print(f"Saved Restored HR image to: {hr_path} ({restored_hr.size[0]}x{restored_hr.size[1]})")

    print("\nDone! Automated test complete.")

if __name__ == "__main__":
    main()
