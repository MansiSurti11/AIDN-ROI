import torch
from PIL import Image
import torchvision.transforms.functional as TF
import math

from base.config import load_cfg_from_cfg_file
from models import get_model
import logging

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

        lr_t = lr_t.clamp(0, 1)
        lr_img = TF.to_pil_image(lr_t.squeeze(0).cpu())

        orig_w, orig_h = orig_size
        lr_w = round(orig_w / scale)
        lr_h = round(orig_h / scale)
        lr_img = lr_img.crop((0, 0, lr_w, lr_h))
        return lr_img

    def restore_patch(
        self,
        lr_image: Image.Image,
        bbox_lr: tuple,
        scale: float
    ) -> Image.Image:
        x, y, w, h = bbox_lr

        lr_w, lr_h = lr_image.size
        x = max(0, min(x, lr_w - 1))
        y = max(0, min(y, lr_h - 1))
        w = min(w, lr_w - x)
        h = min(h, lr_h - y)

        lr_patch = lr_image.crop((x, y, x + w, y + h))

        lr_patch_padded, patch_orig_size = self._pad_to_multiple(lr_patch, multiple=12)

        patch_t = TF.to_tensor(lr_patch_padded).unsqueeze(0).to(self.device)
        scale_t = torch.tensor([scale], dtype=torch.float32).to(self.device)

        patch_w, patch_h = patch_orig_size
        hr_w = round(patch_w * scale)
        hr_h = round(patch_h * scale)

        padded_hr_h = round(patch_t.shape[-2] * scale)
        padded_hr_w = round(patch_t.shape[-1] * scale)
        with torch.no_grad():
            hr_patch_t = self.restore_net(patch_t, scale_t, padded_hr_h, padded_hr_w)

        hr_patch_t = hr_patch_t.clamp(0, 1)
        hr_patch = TF.to_pil_image(hr_patch_t.squeeze(0).cpu())
        hr_patch = hr_patch.crop((0, 0, hr_w, hr_h))

        return hr_patch
