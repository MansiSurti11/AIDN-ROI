# 🔍 AIDN Advanced Feature Guide

Welcome to the enhanced AIDN ROI Selective Restore platform. This guide details the four major features added to provide a premium, high-performance, and secure image restoration experience.

---

## 1. ✨ Auto-Saliency ROI Detection
**Goal:** Automatically identify the most "visually interesting" regions for restoration.

*   **How it works:** Uses a **Spectral Residual Saliency** algorithm (Hou & Zhang, 2007) to analyze the image's frequency domain. It identifies regions that stand out from the background.
*   **How to use:**
    1. After compressing your image, look for the **✨ Auto Propose** button in Step 3.
    2. Click it to let the AI suggest an initial ROI.
    3. The cropper will automatically jump to the most salient object (e.g., a bird, a face, or a high-contrast sign).

---

## 2. ➕ Batched ROI Restoration
**Goal:** Restore multiple regions simultaneously in a single, high-speed inference call.

*   **How it works:** Instead of running the neural network multiple times, we crop all selected regions, pad them into a single 4D Tensor, and process them in one batch.
*   **How to use:**
    1. Draw your first box and click **➕ Add to Batch**.
    2. Note the semi-transparent overlay and ROI number appearing on the image.
    3. Repeat for any other regions you want to restore.
    4. Click **🚀 Restore All (Batch Mode)** in Step 4. All regions will be processed together.

---

## 3. 🔍 Synchronized Interactive Split-Slider
**Goal:** A premium comparison tool with synchronized zoom and pan capabilities.

*   **How it works:** A custom HTML5/JS component provides a side-by-side view with a shared coordinate system.
*   **How to use:**
    1. In the "Restoration Results" section, select the **🔍 Interactive Comparison** tab.
    2. **Zoom:** Use your mouse scroll wheel to zoom into both the LR and HR patches at once.
    3. **Pan:** Click and drag to move both images synchronously.
    4. This allows for precise pixel-level comparison of textures and sharp edges.

---

## 4. 🛡️ Steganographic Watermarking
**Goal:** Hide invisible ownership proof directly inside your compressed LR image.

*   **How it works:** Uses **LSB Steganography** to embed a secret message in the least significant bits of the LR image pixels. It leverages the fact that AIDN already hides info in the LR data, making the watermark exceptionally subtle.
*   **How to use:**
    1. Before clicking "Compress", enter your proof text (e.g., "Copyright © 2026 Admin") in the **©️ Watermarking** sidebar field.
    2. Click **Compress**. The resulting saved/downloaded LR image now contains your hidden text.
    3. **Verification:** When you (or anyone) uploads this LR image back into the tool for restoration, a **🛡️ Watermark Detected** alert will automatically appear if the hidden text is found.

---

### 🚀 Getting Started
To experience all features at once:
1. Upload a high-res image.
2. Enter a watermark in the sidebar.
3. Click **🗜️ Compress**.
4. Use **✨ Auto Propose** to find the first region, then click **➕ Add to Batch**.
5. Manually select a second region and click **➕ Add to Batch**.
6. Click **🚀 Restore All (Batch Mode)**.
7. Switch to the **🔍 Interactive Comparison** tab to see the magic happen!
