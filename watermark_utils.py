import numpy as np
from PIL import Image

def text_to_bits(text: str) -> list:
    """Convert text to a list of bits using UTF-8 encoding."""
    bits = []
    # bytearray for UTF-8 bytes including null terminator
    data = (text + '\0').encode('utf-8')
    for byte in data:
        bin_char = bin(byte)[2:].zfill(8)
        bits.extend([int(b) for b in bin_char])
    return bits

def bits_to_text(bits: list) -> str:
    """Convert a list of bits to text using UTF-8 decoding."""
    bytes_data = []
    for i in range(0, len(bits), 8):
        byte_bits = bits[i:i+8]
        if len(byte_bits) < 8:
            break
        byte_val = int(''.join(map(str, byte_bits)), 2)
        if byte_val == 0:  # Null terminator
            break
        bytes_data.append(byte_val)
    
    try:
        return bytes(bytes_data).decode('utf-8')
    except UnicodeDecodeError:
        return "ERROR: Decoding Failed (Corrupted Data)"

def embed_lsb_watermark(image: Image.Image, text: str) -> Image.Image:
    """
    Embeds text into the least significant bit (LSB) of the image.
    Supports larger data by spreading it across all color channels.
    """
    img_data = np.array(image)
    bits = text_to_bits(text + '\0') # Add null terminator
    
    if len(bits) > img_data.size:
        raise ValueError("Text is too long to be embedded in this image.")
    
    flat_data = img_data.flatten()
    
    # Modify LSB of each pixel value
    # 254 is 11111110 in binary, clears the LSB
    for i in range(len(bits)):
        flat_data[i] = (flat_data[i] & 254) | bits[i]
        
    res_img = Image.fromarray(flat_data.reshape(img_data.shape))
    return res_img

def extract_lsb_watermark(image: Image.Image) -> str:
    """Extracts text from the LSB of the image."""
    img_data = np.array(image)
    flat_data = img_data.flatten()
    
    bits = [flat_data[i] & 1 for i in range(flat_data.size)]
    return bits_to_text(bits)
