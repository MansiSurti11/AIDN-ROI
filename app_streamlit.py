import streamlit as st
from PIL import Image
import numpy as np
import os, io
from aidn_inference import AIDNWrapper
from streamlit_cropper import st_cropper

st.set_page_config(page_title="AIDN ROI Restore", layout="wide")
st.title("🔍 AIDN — Selective ROI Restore")

@st.cache_resource
def load_model():
    return AIDNWrapper(
        config_path='config/DIV2K/AIDN_benchmark.yaml',
        weight_path='LOG/DIV2K/pre-train/model_best.pth'
    )

wrapper = load_model()

with st.sidebar:
    st.header("Settings")
    scale = st.slider("Scale Factor", 1.1, 4.0, 2.0, 0.1)
    st.caption("Higher scale = more compression (smaller LR), harder restoration.")
    st.divider()
    st.markdown("**How to use:**")
    st.markdown("1. Upload an HR image")
    st.markdown("2. Click **Compress**")
    st.markdown("3. **Draw a rectangle** directly on the image with your mouse.")
    st.markdown("4. Click **Restore Selection**")

st.header("Step 1 — Upload HR Image")
uploaded = st.file_uploader("Choose an image", type=["png", "jpg", "jpeg", "webp"])

if uploaded:
    hr_img = Image.open(uploaded).convert("RGB")
    max_dim = 960
    if max(hr_img.size) > max_dim:
        ratio = max_dim / max(hr_img.size)
        hr_img = hr_img.resize((int(hr_img.size[0] * ratio), int(hr_img.size[1] * ratio)), Image.LANCZOS)
        st.warning(f"Image was automatically downscaled to {hr_img.size[0]}×{hr_img.size[1]} px to prevent CPU memory limit exhaustion.")
    st.image(hr_img, caption=f"HR Input — {hr_img.size[0]}×{hr_img.size[1]} px")

    st.header("Step 2 — Compress")
    if st.button("🗜️ Compress with AIDN", type="primary"):
        with st.spinner("Embedding HR information into LR image..."):
            lr_img = wrapper.embed(hr_img, scale)
            st.session_state['lr_img'] = lr_img
            st.session_state['scale'] = scale
        st.success(f"Done! Embedded LR size: {lr_img.size[0]}×{lr_img.size[1]} px")

if 'lr_img' in st.session_state:
    st.header("Step 3 — Select Region of Interest")
    lr_img = st.session_state['lr_img']
    scale_used = st.session_state['scale']
    W, H = lr_img.size

    st.caption("Draw a rectangle on the image to select the region you want to restore in High-Res.")
    
    # st_cropper with return_type='box' gives us the exact ROI coordinates
    # We set aspect_ratio=None to allow any rectangle shape
    rect = st_cropper(
        lr_img, 
        realtime_update=True, 
        box_color='#FF8C00', 
        aspect_ratio=None, 
        return_type='box',
        key="roi_cropper"
    )
    
    if rect:
        # st_cropper returns coords relative to the original PIL image size
        left = int(rect['left'])
        top = int(rect['top'])
        width = int(rect['width'])
        height = int(rect['height'])
        
        # Ensure minimum size for valid restoration
        width = max(width, 16)
        height = max(height, 16)
        
        bbox_lr = (left, top, width, height)
        st.info(f"Selected ROI: {width}x{height} pixels at ({left}, {top})")

    st.header("Step 4 — Restore Selected Region")
    if st.button("✨ Restore HR Patch", type="primary"):
        with st.spinner("Restoring selected region..."):
            hr_patch = wrapper.restore_patch(
                lr_image=lr_img,
                bbox_lr=bbox_lr,
                scale=scale_used
            )

        res_col1, res_col2 = st.columns(2)
        with res_col1:
            lr_crop = lr_img.crop((left, top, left + width, top + height))
            st.image(lr_crop, caption=f"LR patch — {width}×{height} px", use_column_width=True)
        with res_col2:
            st.image(hr_patch, caption=f"Restored HR patch — {hr_patch.size[0]}×{hr_patch.size[1]} px", use_column_width=True)

        buf = io.BytesIO()
        hr_patch.save(buf, format="PNG")
        st.download_button(
            "⬇️ Download Restored Patch",
            data=buf.getvalue(),
            file_name="restored_patch.png",
            mime="image/png"
        )
