import gradio as gr
from PIL import Image
from aidn_inference import AIDNWrapper

wrapper = AIDNWrapper(
    config_path='config/DIV2K/AIDN_benchmark.yaml',
    weight_path='LOG/DIV2K/pre-train/model_best.pth'
)

_embedded_lr: Image.Image = None

def compress(hr_img, scale):
    global _embedded_lr
    if hr_img is None:
        return None, "⚠️ Please upload an image first."
    hr_img_pil = Image.fromarray(hr_img)
    max_dim = 960
    if max(hr_img_pil.size) > max_dim:
        ratio = max_dim / max(hr_img_pil.size)
        hr_img_pil = hr_img_pil.resize((int(hr_img_pil.size[0] * ratio), int(hr_img_pil.size[1] * ratio)), Image.LANCZOS)
        print(f"Warning: Image was downscaled to {hr_img_pil.size} to prevent OOM.")
    _embedded_lr = wrapper.embed(hr_img_pil, scale)
    lr_w, lr_h = _embedded_lr.size
    info = f"✅ Embedded LR size: {lr_w} × {lr_h} px  (scale ×{scale:.1f})"
    return _embedded_lr, info

def restore_roi(scale, patch_w, patch_h, click_data: gr.SelectData):
    global _embedded_lr
    if _embedded_lr is None:
        return None, "⚠️ Compress an image first."

    cx, cy = click_data.index
    x = max(0, cx - patch_w // 2)
    y = max(0, cy - patch_h // 2)

    hr_patch = wrapper.restore_patch(
        lr_image=_embedded_lr,
        bbox_lr=(x, y, patch_w, patch_h),
        scale=scale
    )

    hr_w, hr_h = hr_patch.size
    info = (f"✅ Restored HR patch: {hr_w} × {hr_h} px  "
            f"(from LR region x={x}, y={y}, w={patch_w}, h={patch_h})")
    return hr_patch, info

with gr.Blocks(title="AIDN ROI Restore", theme=gr.themes.Soft()) as demo:
    gr.Markdown("""
    # 🔍 AIDN — Selective ROI Restore
    **Step 1:** Upload a high-resolution image and choose a scale factor, then click **Compress**.  
    **Step 2:** Click anywhere on the embedded LR image to restore just that region in full HR quality.
    """)

    with gr.Row():
        with gr.Column(scale=1):
            hr_input = gr.Image(label="① Upload HR Image", type='numpy')
            scale_slider = gr.Slider(
                minimum=1.1, maximum=4.0, value=2.0, step=0.1,
                label="Scale Factor"
            )
            compress_btn = gr.Button("Compress →", variant="primary")

        with gr.Column(scale=1):
            lr_output = gr.Image(
                label="② Embedded LR  (click to select ROI center)",
                interactive=False
            )
            with gr.Row():
                patch_w = gr.Slider(32, 256, value=96, step=16, label="Patch Width (LR px)")
                patch_h = gr.Slider(32, 256, value=96, step=16, label="Patch Height (LR px)")
            compress_info = gr.Textbox(label="", interactive=False, show_label=False)

        with gr.Column(scale=1):
            hr_patch_output = gr.Image(label="③ Restored HR Patch", interactive=False)
            restore_info = gr.Textbox(label="", interactive=False, show_label=False)

    compress_btn.click(
        fn=compress,
        inputs=[hr_input, scale_slider],
        outputs=[lr_output, compress_info]
    )

    lr_output.select(
        fn=restore_roi,
        inputs=[scale_slider, patch_w, patch_h],
        outputs=[hr_patch_output, restore_info]
    )

    gr.Markdown("""
    ---
    ### Tips
    - Use **AIDN+** weights if you plan to save/share the LR image as JPEG — it is trained to survive lossy compression.
    - The patch size sliders control how large a region you restore (in LR pixel units). The restored HR patch will be `patch_size × scale` pixels.
    - You can click multiple times on different spots without re-compressing.
    """)

if __name__ == "__main__":
    demo.launch(share=False, server_port=7860)
