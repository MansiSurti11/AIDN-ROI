const canvas = document.getElementById('imageCanvas');
const ctx = canvas.getContext('2d');
const hdUpload = document.getElementById('hdUpload');
const encodeBtn = document.getElementById('encodeBtn');
const encodeScaleInput = document.getElementById('encodeScale');
const encodeStatus = document.getElementById('encodeStatus');
const progressContainer = document.getElementById('progressContainer');
const progressBar = document.getElementById('progressBar');
const progressLabel = document.getElementById('progressLabel');

const imageUpload = document.getElementById('imageUpload');
const scaleFactorInput = document.getElementById('scaleFactor');
const restoreBtn = document.getElementById('restoreBtn');
const compareBtn = document.getElementById('compareBtn');
const resetBtn = document.getElementById('resetBtn');
const statusMsg = document.getElementById('statusMsg');

const tabEncode = document.getElementById('tabEncode');
const tabRestore = document.getElementById('tabRestore');
const encodeSection = document.getElementById('encodeSection');
const restoreSection = document.getElementById('restoreSection');

const roiOverlay = document.getElementById('roiOverlay');
const canvasWrapper = document.getElementById('canvasWrapper');
const resultSection = document.getElementById('resultSection');
const comparisonImg = document.getElementById('comparisonImg');
const closeResultBtn = document.getElementById('closeResultBtn');

let originalImage = null;
let imageFile = null;
let hdFile = null;
let startX, startY;
let isDrawing = false;
let bbox = null;
let currentScale = 1.0;
let pollingInterval = null;

const API_BASE = "http://localhost:8000";

// --- Tab Logic ---
tabEncode.addEventListener('click', () => {
    tabEncode.classList.add('active');
    tabRestore.classList.remove('active');
    encodeSection.classList.remove('hidden');
    restoreSection.classList.add('hidden');
});

tabRestore.addEventListener('click', () => {
    tabRestore.classList.add('active');
    tabEncode.classList.remove('active');
    restoreSection.classList.remove('hidden');
    encodeSection.classList.add('hidden');
});

// --- Encoding Logic (Step 1) ---
hdUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;
    hdFile = file;
    
    const reader = new FileReader();
    reader.onload = (event) => {
        const tempImg = new Image();
        tempImg.onload = () => {
            if (Math.max(tempImg.width, tempImg.height) > 2560) {
                encodeStatus.textContent = `High-res image selected: ${tempImg.width}x${tempImg.height}. Note: Image will be safe-resized for memory.`;
            } else {
                encodeStatus.textContent = `High-res image selected: ${tempImg.width}x${tempImg.height}. Ready to encode.`;
            }
        };
        tempImg.src = event.target.result;
    };
    reader.readAsDataURL(file);
    
    encodeBtn.disabled = false;
});

async function pollProgress(taskId) {
    pollingInterval = setInterval(async () => {
        try {
            const response = await fetch(`${API_BASE}/encoding-progress/${taskId}`);
            const data = await response.json();
            const percent = data.progress;
            
            progressBar.style.width = `${percent}%`;
            progressLabel.textContent = `${percent}%`;
            
            if (percent >= 100) {
                clearInterval(pollingInterval);
            }
        } catch (e) {
            console.error("Polling error", e);
        }
    }, 500);
}

encodeBtn.addEventListener('click', async () => {
    if (!hdFile) return;
    
    const taskId = "task_" + Math.random().toString(36).substr(2, 9);
    encodeStatus.textContent = "Encoding image in tiles...";
    encodeBtn.disabled = true;
    progressContainer.classList.remove('hidden');
    progressBar.style.width = "0%";
    progressLabel.textContent = "0%";
    
    // Start polling
    pollProgress(taskId);
    
    const formData = new FormData();
    formData.append('image', hdFile);
    formData.append('scale_factor', encodeScaleInput.value);
    formData.append('task_id', taskId);
    
    try {
        const response = await fetch(`${API_BASE}/embed`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error(await response.text());
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        clearInterval(pollingInterval);
        progressBar.style.width = "100%";
        progressLabel.textContent = "100%";
        
        const newImg = new Image();
        newImg.onload = () => {
            originalImage = newImg;
            imageFile = new File([blob], "embedded_lr.png", { type: "image/png" });
            renderImage();
            
            setTimeout(() => {
                tabRestore.click();
                progressContainer.classList.add('hidden');
                statusMsg.textContent = "Embedded LR image generated. Select an ROI to restore.";
                scaleFactorInput.value = encodeScaleInput.value;
                encodeBtn.disabled = false;
                encodeStatus.textContent = "Encoding complete.";
            }, 1000);
        };
        newImg.src = url;
    } catch (error) {
        clearInterval(pollingInterval);
        progressContainer.classList.add('hidden');
        encodeStatus.textContent = "Error: " + error.message;
        console.error(error);
        encodeBtn.disabled = false;
    }
});

// --- Restoration Logic (Step 2) ---
imageUpload.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    imageFile = file;
    const reader = new FileReader();
    reader.onload = (event) => {
        originalImage = new Image();
        originalImage.onload = () => {
            renderImage();
            statusMsg.textContent = "Image loaded. Click and drag on the image to select an ROI.";
            resetBtn.disabled = false;
            roiOverlay.style.display = "none";
            bbox = null;
            restoreBtn.disabled = true;
            compareBtn.disabled = true;
        };
        originalImage.src = event.target.result;
    };
    reader.readAsDataURL(file);
});

function renderImage() {
    if (!originalImage) return;
    const maxWidth = canvasWrapper.clientWidth;
    const maxHeight = canvasWrapper.clientHeight;
    
    let width = originalImage.width;
    let height = originalImage.height;
    
    currentScale = Math.min(maxWidth / width, 0.8 * window.innerHeight / height, 1.0);
    
    canvas.width = width * currentScale;
    canvas.height = height * currentScale;
    ctx.drawImage(originalImage, 0, 0, canvas.width, canvas.height);
}

// ROI Selection logic
canvas.addEventListener('mousedown', (e) => {
    if (!originalImage || !tabRestore.classList.contains('active')) return;
    isDrawing = true;
    const rect = canvas.getBoundingClientRect();
    startX = e.clientX - rect.left;
    startY = e.clientY - rect.top;
    
    roiOverlay.style.display = "block";
    roiOverlay.style.left = startX + "px";
    roiOverlay.style.top = startY + "px";
    roiOverlay.style.width = "0px";
    roiOverlay.style.height = "0px";
});

window.addEventListener('mousemove', (e) => {
    if (!isDrawing) return;
    const rect = canvas.getBoundingClientRect();
    const currentX = e.clientX - rect.left;
    const currentY = e.clientY - rect.top;
    
    const width = currentX - startX;
    const height = currentY - startY;
    
    roiOverlay.style.left = (width < 0 ? currentX : startX) + "px";
    roiOverlay.style.top = (height < 0 ? currentY : startY) + "px";
    roiOverlay.style.width = Math.abs(width) + "px";
    roiOverlay.style.height = Math.abs(height) + "px";
});

window.addEventListener('mouseup', (e) => {
    if (!isDrawing) return;
    isDrawing = false;
    
    const overlayRect = roiOverlay.getBoundingClientRect();
    const canvasRect = canvas.getBoundingClientRect();
    
    const x = (overlayRect.left - canvasRect.left) / currentScale;
    const y = (overlayRect.top - canvasRect.top) / currentScale;
    const w = overlayRect.width / currentScale;
    const h = overlayRect.height / currentScale;
    
    if (w > 5 && h > 5) {
        bbox = { 
            x: Math.round(x), 
            y: Math.round(y), 
            w: Math.round(w), 
            h: Math.round(h) 
        };
        restoreBtn.disabled = false;
        compareBtn.disabled = false;
        statusMsg.textContent = `ROI selected: ${bbox.w}x${bbox.h}. Ready to restore.`;
    } else {
        roiOverlay.style.display = "none";
        bbox = null;
        restoreBtn.disabled = true;
        compareBtn.disabled = true;
    }
});

restoreBtn.addEventListener('click', async () => {
    if (!bbox || !imageFile) return;
    
    statusMsg.textContent = "Processing ROI restoration...";
    restoreBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('bbox', JSON.stringify(bbox));
    formData.append('scale_factor', scaleFactorInput.value);
    
    try {
        const response = await fetch(`${API_BASE}/restore-roi`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error(await response.text());
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        const newImg = new Image();
        newImg.onload = () => {
            originalImage = newImg;
            imageFile = new File([blob], "restored_composite.png", { type: "image/png" });
            renderImage();
            statusMsg.textContent = "Restoration complete!";
            roiOverlay.style.display = "none";
            bbox = null;
            restoreBtn.disabled = true;
            compareBtn.disabled = true;
        };
        newImg.src = url;
    } catch (error) {
        statusMsg.textContent = "Error: " + error.message;
        console.error(error);
        restoreBtn.disabled = false;
    }
});

compareBtn.addEventListener('click', async () => {
    if (!bbox || !imageFile) return;
    
    statusMsg.textContent = "Generating comparison...";
    compareBtn.disabled = true;
    
    const formData = new FormData();
    formData.append('image', imageFile);
    formData.append('bbox', JSON.stringify(bbox));
    formData.append('scale_factor', scaleFactorInput.value);
    
    try {
        const response = await fetch(`${API_BASE}/compare-roi`, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) throw new Error(await response.text());
        
        const blob = await response.blob();
        const url = URL.createObjectURL(blob);
        
        comparisonImg.src = url;
        resultSection.classList.remove('hidden');
        statusMsg.textContent = "Comparison ready.";
        compareBtn.disabled = false;
    } catch (error) {
        statusMsg.textContent = "Error: " + error.message;
        console.error(error);
        compareBtn.disabled = false;
    }
});

resetBtn.addEventListener('click', () => {
    location.reload();
});

closeResultBtn.addEventListener('click', () => {
    resultSection.classList.add('hidden');
});

window.addEventListener('resize', () => {
    if (originalImage) renderImage();
});
