import torch
from PIL import Image
import torchvision.transforms.functional as TF
import math

from base.config import load_cfg_from_cfg_file
from models import get_model
import logging
from saliency_utils import propose_roi, get_spectral_residual_saliency
from watermark_utils import embed_lsb_watermark, extract_lsb_watermark

class AIDNWrapper:
    """
    Thin wrapper around AIDN's embedding and restoration networks.
    Supports full-image embedding and patch-level selective restoration.
    """

    def __init__(self, config_path: str, weight_path: str, device: str = None):
        if device is None:
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.device = device
        print(f"[AIDN] Using device: {device}")

        cfg = load_cfg_from_cfg_file(config_path)
        self.model = get_model(cfg, logging.getLogger())

        ckpt = torch.load(weight_path, map_location=device)
        self.model.load_state_dict(ckpt['state_dict'], strict=False)
        self.model.eval().to(device)

        self._find_subnets()

    def _find_subnets(self):
        children = dict(self.model.named_children())
        print(f"[AIDN] Sub-networks found: {list(children.keys())}")
        if 'down_net' in children and 'up_net' in children:
            self.embed_net = self.model.down_net
            self.restore_net = self.model.up_net
        elif 'E' in children and 'R' in children:
            self.embed_net = self.model.E
            self.restore_net = self.model.R
        elif 'embed_net' in children and 'restore_net' in children:
            self.embed_net = self.model.embed_net
            self.restore_net = self.model.restore_net
        else:
            self.embed_net = None
            self.restore_net = None
            
        self.quantizer = getattr(self.model, 'quantizer', None)

    def _pad_to_multiple(self, img: Image.Image, multiple: int = 12):
        w, h = img.size
        new_w = math.ceil(w / multiple) * multiple
        new_h = math.ceil(h / multiple) * multiple
        if new_w == w and new_h == h:
            return img, (w, h)
        padded = Image.new(img.mode, (new_w, new_h), 0)
        padded.paste(img, (0, 0))
        return padded, (w, h)

    def embed(self, hr_image: Image.Image, scale: float) -> Image.Image:
        hr_padded, orig_size = self._pad_to_multiple(hr_image, multiple=12)

        x = TF.to_tensor(hr_padded).unsqueeze(0).to(self.device)
        scale_t = torch.tensor([1.0/scale], dtype=torch.float32).to(self.device)

        with torch.no_grad():
            lr_t = self.embed_net(x, scale_t)
            if self.quantizer is not None:
                lr_t = self.quantizer(lr_t)

        lr_t = lr_t.clamp(0, 1)
        lr_img = TF.to_pil_image(lr_t.squeeze(0).cpu())

        orig_w, orig_h = orig_size
        lr_w = round(orig_w / scale)
        lr_h = round(orig_h / scale)
        lr_img = lr_img.crop((0, 0, lr_w, lr_h))
        return lr_img

    def embed_watermark(self, lr_image: Image.Image, text: str) -> Image.Image:
        """Embeds a digital watermark into the LR image."""
        return embed_lsb_watermark(lr_image, text)

    def verify_watermark(self, lr_image: Image.Image) -> str:
        """Extracts a digital watermark from the LR image."""
        return extract_lsb_watermark(lr_image)

    def restore_patch(
        self,
        lr_image: Image.Image,
        bbox_lr: tuple,
        scale: float
    ) -> Image.Image:
        """
        Restores a single patch from the LR image.
        Uses single-batch inference.
        """
        return self.restore_patches_batch(lr_image, [bbox_lr], scale)[0]

    def restore_patches_batch(
        self,
        lr_image: Image.Image,
        bboxes_lr: list,
        scale: float
    ) -> list:
        """
        Restores multiple patches in a single batched inference call.
        bboxes_lr: list of (x, y, w, h)
        """
        if not bboxes_lr:
            return []

        patches = []
        orig_sizes = []
        
        # 1. Prepare patches and find max dimensions
        max_w, max_h = 0, 0
        for bbox in bboxes_lr:
            x, y, w, h = bbox
            # Safety checks
            lr_w_img, lr_h_img = lr_image.size
            x = max(0, min(x, lr_w_img - 1))
            y = max(0, min(y, lr_h_img - 1))
            w = min(w, lr_w_img - x)
            h = min(h, lr_h_img - y)
            
            patch = lr_image.crop((x, y, x + w, y + h))
            patches.append(patch)
            orig_sizes.append((w, h))
            max_w = max(max_w, w)
            max_h = max(max_h, h)

        # 2. Pad all patches to the same max dimensions (multiple of 12)
        multiple = 12
        target_w = math.ceil(max_w / multiple) * multiple
        target_h = math.ceil(max_h / multiple) * multiple
        
        padded_tensors = []
        for patch in patches:
            # We don't use _pad_to_multiple here because we want ALL to be the SAME size
            # Create a black background of target size
            full_padded = Image.new(patch.mode, (target_w, target_h), 0)
            full_padded.paste(patch, (0, 0))
            padded_tensors.append(TF.to_tensor(full_padded))

        # 3. Stack into a single 4D tensor: [N, 3, H, W]
        batch_t = torch.stack(padded_tensors).to(self.device)
        scale_t = torch.tensor([scale], dtype=torch.float32).to(self.device)

        # 4. Batched Inference
        hr_h_padded = round(target_h * scale)
        hr_w_padded = round(target_w * scale)
        
        with torch.no_grad():
            # self.restore_net supports batch size N
            hr_batch_t = self.restore_net(batch_t, scale_t, hr_h_padded, hr_w_padded)

        hr_batch_t = hr_batch_t.clamp(0, 1)

        # 5. Convert back to list of PIL images and crop to original HR sizes
        results = []
        for i, (orig_w, orig_h) in enumerate(orig_sizes):
            hr_patch = TF.to_pil_image(hr_batch_t[i].cpu())
            # Real HR size
            hr_w = round(orig_w * scale)
            hr_h = round(orig_h * scale)
            hr_patch = hr_patch.crop((0, 0, hr_w, hr_h))
            results.append(hr_patch)

        return results

    def restore_full_image(
        self,
        lr_image: Image.Image,
        scale: float,
        patch_size: int = 128,
        overlap: int = 16
    ) -> Image.Image:
        """
        Restores a large LR image by tiling it into patches, processing each,
        and stitching them back together. Prevents OOM for high-res inputs.
        """
        lr_w, lr_h = lr_image.size
        hr_w, hr_h = round(lr_w * scale), round(lr_h * scale)
        
        # Final HR canvas
        full_hr = Image.new("RGB", (hr_w, hr_h))
        
        stride = patch_size - overlap
        
        # Progress tracking (optional)
        num_tiles_x = math.ceil(lr_w / stride)
        num_tiles_y = math.ceil(lr_h / stride)
        print(f"[AIDN] Tiled restoration: {num_tiles_x}x{num_tiles_y} grid...")

        for y in range(0, lr_h, stride):
            for x in range(0, lr_w, stride):
                # 1. Define patch bbox with safety boundaries
                cur_w = min(patch_size, lr_w - x)
                cur_h = min(patch_size, lr_h - y)
                
                # 2. Restore this patch
                hr_patch = self.restore_patch(lr_image, (x, y, cur_w, cur_h), scale)
                
                # 3. Stitching logic
                # To avoid edge artifacts, we only paste the "center" part of the patch
                # unless we are at the very edges of the whole image.
                p_left = overlap // 2 if x > 0 else 0
                p_top = overlap // 2 if y > 0 else 0
                p_right = hr_patch.size[0] - (overlap // 2) if (x + cur_w) < lr_w else hr_patch.size[0]
                p_bottom = hr_patch.size[1] - (overlap // 2) if (y + cur_h) < lr_h else hr_patch.size[1]
                
                # Crop the unique center part
                unique_patch = hr_patch.crop((p_left, p_top, p_right, p_bottom))
                
                # Paste at the correct HR location
                full_hr.paste(unique_patch, (round((x + p_left) * scale), round((y + p_top) * scale)))
        
        return full_hr

    def get_saliency_roi(self, image: Image.Image, box_size: int = 256):
        """
        Computes saliency map and proposes a bounding box ROI.
        """
        # Suggest a square ROI for consistency
        return propose_roi(image, box_width=box_size, box_height=box_size)
