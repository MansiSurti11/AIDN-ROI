import streamlit as st
from PIL import Image
import numpy as np
import os, io
from aidn_inference import AIDNWrapper
from streamlit_cropper import st_cropper
from PIL import ImageDraw
import base64
import streamlit.components.v1 as components
from streamlit_image_comparison import image_comparison

# --- PAGE CONFIG ---
st.set_page_config(page_title="AIDN ROI Restore", layout="wide", initial_sidebar_state="expanded")

# --- CUSTOM CSS ---
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .stButton>button { width: 100%; border-radius: 8px; }
    .stMetric { background-color: #1e2130; padding: 10px; border-radius: 10px; border: 1px solid #333; }
    .step-header { color: #FF8C00; font-weight: bold; margin-bottom: 20px; text-transform: uppercase; letter-spacing: 1.5px; border-bottom: 2px solid #FF8C00; padding-bottom: 5px; }
    .card { background: #1a1c24; padding: 20px; border-radius: 12px; border: 1px solid #2d2d2d; margin-bottom: 20px; }
</style>
""", unsafe_allow_html=True)

# --- UTILS ---
@st.cache_resource
def load_model():
    return AIDNWrapper(
        config_path='config/DIV2K/AIDN_benchmark.yaml',
        weight_path='LOG/DIV2K/pre-train/model_best.pth'
    )

def get_image_base64(img):
    buffered = io.BytesIO()
    img.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode()

def sync_zoom_compare(img1, img2, label1="LR Original", label2="AIDN Restored"):
    """Industry-standard synchronized zoom/pan using OpenSeadragon with polished UI."""
    b1 = get_image_base64(img1)
    b2 = get_image_base64(img2)
    
    html_code = f"""
    <div id="osd-container" style="display: flex; width: 100%; height: 700px; background: #000; border-radius: 12px; overflow: hidden; border: 1px solid #333; position: relative;">
        <div style="width: 50%; height: 100%; border-right: 1px solid #444; position: relative;">
            <div id="viewer1" style="width: 100%; height: 100%;"></div>
            <div style="position: absolute; top: 10px; left: 10px; background: rgba(0,0,0,0.7); color: #fff; padding: 3px 8px; border-radius: 4px; font-family: sans-serif; font-size: 10px; z-index: 10; font-weight: 600;">{label1}</div>
        </div>
        <div style="width: 50%; height: 100%; position: relative;">
            <div id="viewer2" style="width: 100%; height: 100%;"></div>
            <div style="position: absolute; top: 10px; left: 10px; background: rgba(255,140,0,0.9); color: #fff; padding: 3px 8px; border-radius: 4px; font-family: sans-serif; font-size: 10px; z-index: 10; font-weight: 600;">{label2}</div>
        </div>
        <div style="position: absolute; bottom: 12px; right: 12px; background: rgba(0,0,0,0.5); color: #aaa; padding: 4px 10px; border-radius: 20px; font-family: sans-serif; font-size: 10px; z-index: 10; backdrop-filter: blur(4px);">
            🖱️ Scroll to Sync-Zoom | 🖐️ Drag to Sync-Pan
        </div>
    </div>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/openseadragon.min.js"></script>
    <script>
    (function() {{
        const viewer1 = OpenSeadragon({{
            id: "viewer1",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
            tileSources: {{ type: 'image', url: 'data:image/png;base64,{b1}' }},
            showNavigationControl: false,
            gestureSettingsMouse: {{ clickToZoom: false, scrollZoom: true }},
            pointerDelta: 0,
            imageSmoothingEnabled: false,
            zoomPerScroll: 1.5,
            minZoomLevel: 0.5,
            maxZoomLevel: 100
        }});
        const viewer2 = OpenSeadragon({{
            id: "viewer2",
            prefixUrl: "https://cdnjs.cloudflare.com/ajax/libs/openseadragon/4.1.0/images/",
            tileSources: {{ type: 'image', url: 'data:image/png;base64,{b2}' }},
            showNavigationControl: false,
            gestureSettingsMouse: {{ clickToZoom: false, scrollZoom: true }},
            pointerDelta: 0,
            zoomPerScroll: 1.5,
            minZoomLevel: 0.5,
            maxZoomLevel: 100
        }});
        let isSyncing = false;
        function sync(source, target) {{
            source.addHandler('viewport-change', function() {{
                if (isSyncing) return;
                isSyncing = true;
                target.viewport.zoomTo(source.viewport.getZoom());
                target.viewport.panTo(source.viewport.getCenter());
                isSyncing = false;
            }});
        }}
        viewer1.addHandler('open', () => {{ viewer1.viewport.goHome(true); }});
        viewer2.addHandler('open', () => {{ viewer2.viewport.goHome(true); }});
        sync(viewer1, viewer2);
        sync(viewer2, viewer1);
    }})();
    </script>
    """
    components.html(html_code, height=720, scrolling=False)

def slider_compare_lib(img_lr, img_hr, label_lr="Original (LR)", label_hr="Restored (HR)"):
    """Forced Full-Width Juxtapose Slider using raw HTML/JS for maximum reliability."""
    b_lr = get_image_base64(img_lr)
    b_hr = get_image_base64(img_hr)
    
    # Force high-vis labels and 100% scaling
    html_code = f"""
    <link rel="stylesheet" href="https://cdn.knightlab.com/libs/juxtapose/latest/css/juxtapose.css">
    <div style="width: 100%; height: 800px; overflow: hidden; border-radius: 12px; border: 1px solid #333;">
        <div id="juxtapose-embed" style="width: 100%; height: 100%;"></div>
    </div>
    <script src="https://cdn.knightlab.com/libs/juxtapose/latest/js/juxtapose.min.js"></script>
    <script>
    (function() {{
        new juxtapose.JXSlider('#juxtapose-embed', [
            {{
                src: 'data:image/png;base64,{b_hr}',
                label: '{label_hr}'
            }},
            {{
                src: 'data:image/png;base64,{b_lr}',
                label: '{label_lr}'
            }}
        ], {{
            animate: true,
            showLabels: true,
            showCredits: false,
            startingPosition: "50%",
            makeResponsive: true
        }});
    }})();
    </script>
    """
    # Use standard components.html but force it to take full width
    components.html(html_code, height=820, scrolling=False)

# --- INITIALIZATION ---
wrapper = load_model()
if 'step' not in st.session_state:
    st.session_state.step = 1
if 'batch_queue' not in st.session_state:
    st.session_state.batch_queue = []

# --- SIDEBAR: STATUS DASHBOARD ---
with st.sidebar:
    st.title("AIDN Dashboard")
    st.divider()
    st.metric("Workflow Stage", f"Step {st.session_state.step}")
    
    if 'hr_img' in st.session_state:
        st.success("✅ Source Image Ready")
        st.caption(f"Dim: {st.session_state.hr_img.size[0]}x{st.session_state.hr_img.size[1]}px")
    else:
        st.info("⌛ Waiting for Source")

    if 'lr_img' in st.session_state:
        st.success("✅ AIDN LR Embedded")
        if st.session_state.get('detected_watermark'):
            st.info(f"🛡️ Ownership: '{st.session_state.detected_watermark}'")
    
    st.divider()
    st.subheader(f"Batch Queue ({len(st.session_state.batch_queue)})")
    if st.session_state.batch_queue:
        for i, bbox in enumerate(st.session_state.batch_queue):
            st.caption(f"#{i+1}: {bbox[2]}x{bbox[3]} at ({bbox[0]},{bbox[1]})")
        if st.button("🗑️ Clear Batch", use_container_width=True):
            st.session_state.batch_queue = []
            st.rerun()
    else:
        st.write("No regions selected yet.")
    
    st.divider()
    st.caption("AIDN v2.1.0 | Stable Library Release")

# --- MAIN WORKFLOW STEPPER ---
col_step1, col_step2, col_step3, col_step4 = st.columns(4)

# Determine button coloring based on current step
def get_step_style(target_step):
    return "primary" if st.session_state.step == target_step else "secondary"

if col_step1.button("📤 1. SOURCE", type=get_step_style(1)):
    st.session_state.step = 1
    st.rerun()
if col_step2.button("🗜️ 2. EMBED", type=get_step_style(2), disabled='hr_img' not in st.session_state):
    st.session_state.step = 2
    st.rerun()
if col_step3.button("🎯 3. TARGET", type=get_step_style(3), disabled='lr_img' not in st.session_state):
    st.session_state.step = 3
    st.rerun()
if col_step4.button("✨ 4. REVIEW", type=get_step_style(4), disabled='last_results' not in st.session_state):
    st.session_state.step = 4
    st.rerun()

st.divider()

# --- STAGE 1: SOURCE ---
if st.session_state.step == 1:
    st.markdown('<div class="step-header">Source Image Acquisition</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        uploaded = st.file_uploader("Upload High-Res (HR) Image", type=["png", "jpg", "jpeg", "webp"])
        if uploaded:
            hr_img = Image.open(uploaded).convert("RGB")
            max_dim = 960
            if max(hr_img.size) > max_dim:
                ratio = max_dim / max(hr_img.size)
                hr_img = hr_img.resize((int(hr_img.size[0] * ratio), int(hr_img.size[1] * ratio)), Image.LANCZOS)
                st.warning(f"Resized to {hr_img.size[0]}x{hr_img.size[1]}px for processing.")
            
            st.session_state.hr_img = hr_img
            st.image(hr_img, caption="Loaded HR Source")
            
            if st.button("🚀 Proceed to Embedding", type="primary"):
                st.session_state.step = 2
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- STAGE 2: EMBED ---
elif st.session_state.step == 2:
    st.markdown('<div class="step-header">AIDN Compression & Steganography</div>', unsafe_allow_html=True)
    with st.container():
        st.markdown('<div class="card">', unsafe_allow_html=True)
        c1, c2 = st.columns(2)
        with c1:
            scale = st.slider("Compression Scale", 1.2, 4.0, 2.0, 0.1)
            st.caption("Higher scale = Smaller LR but harder restoration.")
        with c2:
            wm_text = st.text_input("Steganographic Watermark", "AIDN Copyright 2026")
            st.caption("Hidden ownership proof embedded in LR original.")
        
        if st.button("🗜️ Execute AIDN Compression", type="primary"):
            with st.status("Performing Invertible Downscaling...", expanded=True) as status:
                lr_img = wrapper.embed(st.session_state.hr_img, scale)
                if wm_text:
                    lr_img = wrapper.embed_watermark(lr_img, wm_text)
                st.session_state.lr_img = lr_img
                st.session_state.scale = scale
                st.session_state.detected_watermark = wrapper.verify_watermark(lr_img)
                status.update(label="Compression Complete!", state="complete")
            st.rerun()
            
        if 'lr_img' in st.session_state:
            st.image(st.session_state.lr_img, caption=f"Embedded LR image ({st.session_state.lr_img.size[0]}x{st.session_state.lr_img.size[1]})")
            st.success("Successfully compressed. Information correctly embedded.")
            if st.button("🎯 Proceed to Region Selection", type="primary"):
                st.session_state.step = 3
                st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- STAGE 3: TARGET ---
elif st.session_state.step == 3:
    st.markdown('<div class="step-header">Region of Interest Selection</div>', unsafe_allow_html=True)
    lr_display = st.session_state.lr_img.copy()
    draw = ImageDraw.Draw(lr_display, "RGBA")
    for i, bbox in enumerate(st.session_state.batch_queue):
        lx, ly, lw, lh = bbox
        draw.rectangle([lx, ly, lx + lw, ly + lh], outline="#FF8C00", width=2, fill=(255, 140, 0, 40))
    
    col_sel, col_ctrl = st.columns([3, 1])
    with col_sel:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        rect = st_cropper(lr_display, realtime_update=True, box_color='#FF8C00', return_type='box', 
                          default_coords=st.session_state.get('proposed_roi'), key="cropper")
        st.markdown('</div>', unsafe_allow_html=True)
        
    with col_ctrl:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        st.write("**Selection Controls**")
        if st.button("✨ Auto-Detect ROI"):
            st.session_state.proposed_roi = wrapper.get_saliency_roi(st.session_state.lr_img)
            # Reformat (x, y, w, h) -> (xl, xr, yt, yb) for st_cropper
            x, y, w, h = st.session_state.proposed_roi
            st.session_state.proposed_roi = (x, x+w, y, y+h)
            st.rerun()
            
        if rect:
            curr_bbox = (int(rect['left']), int(rect['top']), int(rect['width']), int(rect['height']))
            if st.button("➕ Add Selection to Batch", type="primary"):
                st.session_state.batch_queue.append(curr_bbox)
                st.toast("Region Added!")
                st.rerun()
        
        st.divider()
        st.write(f"In Batch: **{len(st.session_state.batch_queue)}**")
        if st.button("🚀 Restore Batch & Review", type="primary", disabled=not st.session_state.batch_queue):
            with st.status("Restoring Selected Regions...", expanded=True) as status:
                st.session_state.last_results = wrapper.restore_patches_batch(
                    st.session_state.lr_img, st.session_state.batch_queue, st.session_state.scale)
                st.session_state.last_bboxes = list(st.session_state.batch_queue)
                status.update(label="Restoration Complete!", state="complete")
            st.session_state.step = 4
            st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)

# --- STAGE 4: REVIEW ---
elif st.session_state.step == 4:
    st.markdown('<div class="step-header">High-Resolution Restoration Results</div>', unsafe_allow_html=True)
    
    if 'last_results' in st.session_state:
        for i, (hr_patch, bbox) in enumerate(zip(st.session_state.last_results, st.session_state.last_bboxes)):
            st.divider()
            st.subheader(f"Region #{i+1} — {bbox[2]}x{bbox[3]} at ({bbox[0]},{bbox[1]})")
            
            # Switch to horizontal radio for mode selection - wider than tabs
            view_mode = st.radio(f"Inspector Mode (Region #{i+1})", 
                                 ["🔍 Sync Pro-Inspector", "↔️ Full-Width Slider"], 
                                 horizontal=True, index=1, key=f"mode_{i}")

            lr_crop = st.session_state.lr_img.crop((bbox[0], bbox[1], bbox[0]+bbox[2], bbox[1]+bbox[3]))
            
            if "Slider" in view_mode:
                # Forced large upscale for the slider to ensure it spans the column
                target_w = 1920
                w, h = hr_patch.size
                ratio = target_w / w
                hr_up = hr_patch.resize((target_w, int(h * ratio)), Image.LANCZOS)
                lr_up = lr_crop.resize((target_w, int(h * ratio)), Image.NEAREST)
                slider_compare_lib(lr_up, hr_up)
            else:
                sync_zoom_compare(lr_crop, hr_patch)
            
            buf = io.BytesIO()
            hr_patch.save(buf, format="PNG")
            st.download_button(f"⬇️ Download ROI #{i+1}", buf.getvalue(), f"roi_{i+1}.png", "image/png", key=f"dl_{i}")
    
    st.divider()
    if st.button("⏮️ Back to Selection Stage", use_container_width=True):
        st.session_state.step = 3
        st.rerun()
