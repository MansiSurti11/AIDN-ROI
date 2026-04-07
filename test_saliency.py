import cv2
import numpy as np
from PIL import Image
from saliency_utils import get_spectral_residual_saliency, propose_roi
import os

def test_saliency():
    # Create a dummy image with a bright circle (salient object)
    img_size = 512
    img = np.zeros((img_size, img_size, 3), dtype=np.uint8)
    # Background is dark gray
    img.fill(50)
    # Salient object: bright white circle
    center = (300, 200)
    radius = 40
    cv2.circle(img, center, radius, (255, 255, 255), -1)
    
    pil_img = Image.fromarray(img)
    
    print("Computing saliency map...")
    smap = get_spectral_residual_saliency(pil_img)
    
    # Save for visual inspection
    smap_uint8 = (smap * 255).astype(np.uint8)
    os.makedirs("output", exist_ok=True)
    cv2.imwrite("output/test_saliency_map.png", smap_uint8)
    print("Saliency map saved to output/test_saliency_map.png")
    
    print("Proposing ROI...")
    x, y, w, h = propose_roi(pil_img, 100, 100)
    print(f"Proposed ROI: ({x}, {y}, {w}, {h})")
    
    # Check if the proposed ROI contains the circle center
    # circle center is (300, 200)
    contains_center = (x <= 300 <= x+w) and (y <= 200 <= y+h)
    
    if contains_center:
        print("SUCCESS: Proposed ROI contains the salient object!")
    else:
        print(f"FAILURE: Proposed ROI ({x}, {y}) to ({x+w}, {y+h}) missed the object at {center}")

if __name__ == "__main__":
    test_saliency()
