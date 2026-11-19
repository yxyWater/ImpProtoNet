"""Deterministic image preprocessing for eyelid clinical photographs."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

import torch
from PIL import Image, ImageOps
from torchvision.transforms import functional as functional
import torchvision.transforms as T


IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
IMAGENET_MEAN_RGB = tuple(round(value * 255) for value in IMAGENET_MEAN)


@dataclass(frozen=True)
class FitPadToSquare:
    """Preserve the complete field of view while fitting an image to a square."""

    size: int = 224
    fill: tuple[int, int, int] = IMAGENET_MEAN_RGB

    def __call__(self, image: Image.Image) -> Image.Image:
        image = image.convert("RGB")
        width, height = image.size
        if width < 1 or height < 1:
            raise ValueError(f"Invalid image size: {image.size}")
        scale = min(self.size / width, self.size / height)
        resized_width = max(1, min(self.size, round(width * scale)))
        resized_height = max(1, min(self.size, round(height * scale)))
        resized = image.resize(
            (resized_width, resized_height),
            resample=Image.Resampling.BILINEAR,
        )
        left = (self.size - resized_width) // 2
        top = (self.size - resized_height) // 2
        canvas = Image.new("RGB", (self.size, self.size), self.fill)
        canvas.paste(resized, (left, top))
        return canvas


def deterministic_horizontal_flip(
    probability: float,
    seed: int,
    *identity_parts: object,
) -> bool:
    """Choose a reproducible flip without relying on process-global RNG state."""

    if not 0.0 <= probability <= 1.0:
        raise ValueError("Horizontal flip probability must be between zero and one")
    material = "|".join([str(seed), *(str(item) for item in identity_parts)])
    value = int.from_bytes(
        hashlib.sha256(material.encode("utf-8")).digest()[:8], "big"
    )
    return value / 2**64 < probability


@dataclass(frozen=True)
class ImagePreprocessor:
    """Apply the approved geometry-preserving ResNet18 preprocessing."""

    image_size: int = 224
    mean: tuple[float, float, float] = IMAGENET_MEAN
    std: tuple[float, float, float] = IMAGENET_STD

    def __call__(
        self,
        image: Image.Image,
        *,
        horizontal_flip: bool = False,
        cj_params=None,
        rotation_deg=0.0,
    ) -> torch.Tensor:
        image = image.convert("RGB")
        if horizontal_flip:
            image = ImageOps.mirror(image)
        if rotation_deg != 0.0:
            image = image.rotate(rotation_deg, resample=Image.Resampling.BILINEAR, expand=False, fillcolor=tuple(IMAGENET_MEAN_RGB))
        image = FitPadToSquare(self.image_size)(image)
        tensor = functional.pil_to_tensor(image).to(dtype=torch.float32).div_(255.0)
        if cj_params is not None:
            b, c, s, hue = cj_params
            tensor = functional.adjust_brightness(tensor, b)
            tensor = functional.adjust_contrast(tensor, c)
            tensor = functional.adjust_saturation(tensor, s)
            tensor = functional.adjust_hue(tensor, hue)
        return functional.normalize(tensor, self.mean, self.std)

def deterministic_color_jitter(strength, seed, *identity_parts):
    """Deterministic brightness/contrast/saturation/hue jitter values."""
    import hashlib
    if strength <= 0.0:
        return (1.0, 1.0, 1.0, 0.0)
    material = "|".join([str(seed), *(str(item) for item in identity_parts)])
    h = hashlib.sha256(material.encode("utf-8")).digest()
    b = 1.0 + (int.from_bytes(h[0:4], "big") / 2**32 - 0.5) * 2 * strength
    c = 1.0 + (int.from_bytes(h[4:8], "big") / 2**32 - 0.5) * 2 * strength
    s = 1.0 + (int.from_bytes(h[8:12], "big") / 2**32 - 0.5) * 2 * strength
    hue = (int.from_bytes(h[12:16], "big") / 2**32 - 0.5) * 2 * min(strength, 0.5)
    return (max(0.1, b), max(0.1, c), max(0.1, s), hue)


def deterministic_rotation(max_degrees, seed, *identity_parts):
    """Deterministic rotation angle in degrees."""
    import hashlib
    if max_degrees <= 0.0:
        return 0.0
    material = "|".join([str(seed), *(str(item) for item in identity_parts)])
    h = hashlib.sha256(material.encode("utf-8")).digest()
    angle = (int.from_bytes(h[0:8], "big") / 2**64 - 0.5) * 2 * max_degrees
    return angle
