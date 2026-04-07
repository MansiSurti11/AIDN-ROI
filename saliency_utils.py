import cv2
import numpy as np
from PIL import Image

def get_spectral_residual_saliency(image: Image.Image):
    """
    Computes a saliency map using the Spectral Residual approach (Hou & Zhang, 2007).
    This method is extremely fast and effective for finding unique/unusual objects.
    """
    # 1. Preprocess: Resize and Grayscale
    img = np.array(image.convert("L"))
    img_float = img.astype(np.float32) / 255.0
    
    # 2. Compute FFT
    dft = cv2.dft(img_float, flags=cv2.DFT_COMPLEX_OUTPUT)
    dft_shift = np.fft.fftshift(dft)
    
    # 3. Compute Log Amplitude and Phase
    magnitude, phase = cv2.cartToPolar(dft_shift[:, :, 0], dft_shift[:, :, 1])
    log_amplitude = np.log(magnitude + 1e-6)
    
    # 4. Compute Spectral Residual (Amplitude - Averaged Amplitude)
    # Using a simple box filter as the local average
    avg_log_amplitude = cv2.blur(log_amplitude, (3, 3))
    spectral_residual = log_amplitude - avg_log_amplitude
    
    # 5. Inverse FFT with original phase
    res_mag = np.exp(spectral_residual)
    x, y = cv2.polarToCart(res_mag, phase)
    
    # Reconstruct complex DFT
    recon_dft_shift = np.stack([x, y], axis=-1)
    recon_dft = np.fft.ifftshift(recon_dft_shift)
    res_img = cv2.idft(recon_dft)
    
    # Compute magnitude of reconstruction
    saliency_map = cv2.magnitude(res_img[:, :, 0], res_img[:, :, 1])
    
    # 6. Post-process: Square and Blur
    saliency_map = np.square(saliency_map)
    saliency_map = cv2.GaussianBlur(saliency_map, (9, 9), 2.5)
    
    # Normalize to [0, 1]
    cv2.normalize(saliency_map, saliency_map, 0, 1, cv2.NORM_MINMAX)
    
    return saliency_map

def propose_roi(image: Image.Image, box_width: int = 256, box_height: int = 256):
    """
    Finds the peak of the saliency map and returns a bounding box (x, y, w, h).
    """
    saliency_map = get_spectral_residual_saliency(image)
    
    # Find the peak coordinates
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(saliency_map)
    center_x, center_y = max_loc
    
    # Define ROI centered at peak
    W, H = image.size
    left = max(0, min(center_x - box_width // 2, W - box_width))
    top = max(0, min(center_y - box_height // 2, H - box_height))
    
    # Final clamping and handling cases where image is smaller than box
    actual_w = min(box_width, W)
    actual_h = min(box_height, H)
    
    return (int(left), int(top), int(actual_w), int(actual_h))
