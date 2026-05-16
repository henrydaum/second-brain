"""Classic DeepDream-style image service with a deterministic fallback."""

from __future__ import annotations

import logging, random

from PIL import Image, ImageChops, ImageEnhance, ImageFilter, ImageOps

from plugins.BaseService import BaseService

logger = logging.getLogger("DeepDream")


class DeepDreamService(BaseService):
    model_name = "deep_dream"
    config_settings = [
        ("Deep Dream Max Size", "deep_dream_max_size", "Largest side rendered by the CNN dream pass.", 768, {"type": "integer"}),
        ("Deep Dream Device", "deep_dream_device", "Device preference: auto, cpu, mps, or cuda.", "auto", {"type": "text"}),
    ]

    def __init__(self, config=None):
        super().__init__(); self.config = config or {}; self.torch = self.model = self.device = None

    def _load(self) -> bool:
        try:
            import torch
            from torchvision import models
            pref = str(self.config.get("deep_dream_device") or "auto")
            self.device = "cuda" if pref == "auto" and torch.cuda.is_available() else "mps" if pref == "auto" and getattr(torch.backends, "mps", None) and torch.backends.mps.is_available() else pref
            if self.device not in {"cpu", "cuda", "mps"}: self.device = "cpu"
            self.torch = torch
            try:
                self.model = models.squeezenet1_1(weights=models.SqueezeNet1_1_Weights.DEFAULT).features.to(self.device).eval()
            except Exception as e:
                logger.warning("Pretrained SqueezeNet unavailable, using untrained CNN: %s", e)
                self.model = models.squeezenet1_1(weights=None).features.to(self.device).eval()
            for p in self.model.parameters(): p.requires_grad_(False)
            self.loaded = True
            return True
        except Exception as e:
            logger.warning("CNN DeepDream unavailable, using fallback: %s", e)
            self.loaded = False
            return False

    def unload(self):
        self.model = None; self.loaded = False
        try:
            if self.torch and self.torch.cuda.is_available(): self.torch.cuda.empty_cache()
        except Exception:
            pass

    def dream(self, image, seed=1, preset="classic", strength=.55, octaves=2, steps=8):
        try:
            if self.loaded and self.model is not None:
                return self._cnn(image, seed, preset, strength, octaves, steps), {"engine": "cnn", "preset": preset}
        except Exception as e:
            logger.warning("CNN dream failed, falling back: %s", e)
        return fallback_dream(image, seed, preset, strength), {"engine": "fallback", "preset": preset}

    def _cnn(self, image, seed, preset, strength, octaves, steps):
        torch = self.torch; random.seed(seed); size = int(self.config.get("deep_dream_max_size") or 768)
        img = image.convert("RGB"); img.thumbnail((size, size))
        to_t = lambda im: torch.tensor(list(im.getdata()), dtype=torch.float32, device=self.device).view(im.height, im.width, 3).permute(2,0,1)[None] / 255
        x = to_t(img); x = (x - .5) / .25
        layers = {"classic": 8, "organic": 10, "crystal": 5, "machine": 12, "soft": 7}
        target = min(layers.get(preset, 16), len(self.model) - 1)
        for octave in range(max(1, min(3, int(octaves)))):
            x = torch.nn.functional.interpolate(x, scale_factor=1.0 + octave * .18, mode="bilinear", align_corners=False).detach()
            for _ in range(max(1, min(14, int(steps)))):
                x.requires_grad_(True); y = x
                for i, layer in enumerate(self.model):
                    y = layer(y)
                    if i >= target: break
                loss = y.square().mean(); loss.backward()
                g = x.grad; x = (x + (float(strength) * .018) * g / (g.abs().mean() + 1e-6)).detach().clamp(-2, 2)
        x = (x * .25 + .5).clamp(0, 1)[0].permute(1,2,0).detach().cpu().numpy()
        return Image.fromarray((x * 255).astype("uint8"), "RGB").resize(image.size, Image.Resampling.LANCZOS)


def fallback_dream(image, seed=1, preset="classic", strength=.55):
    r = random.Random(f"{seed}:{preset}"); img = image.convert("RGB")
    edge = ImageOps.autocontrast(img.filter(ImageFilter.FIND_EDGES)).filter(ImageFilter.GaussianBlur(1.2))
    swirl = ImageChops.offset(img.filter(ImageFilter.SMOOTH_MORE), r.randrange(-18, 19), r.randrange(-18, 19))
    glow = ImageEnhance.Color(ImageChops.screen(img, edge.convert("RGB"))).enhance(1 + float(strength) * .7)
    if preset in {"organic", "soft"}: glow = ImageChops.screen(glow, swirl.filter(ImageFilter.GaussianBlur(2)))
    if preset in {"crystal", "machine"}: glow = ImageEnhance.Sharpness(glow).enhance(1.8)
    return Image.blend(img, glow, max(.15, min(.85, float(strength))))


def build_services(config: dict) -> dict:
    return {"deep_dream": DeepDreamService(config)}
