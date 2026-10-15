"""Sanity-checks the PyTorch models and the converted Core ML packages (run on macOS).

Fails (non-zero exit) when:
  * a PyTorch model is far worse than bicubic on a synthetic line-art image
    (wrong weights / transposed matrices), or
  * a Core ML package disagrees with its PyTorch source.
"""

import argparse
import math
import os
import sys

import coremltools as ct
import numpy as np
import torch
from PIL import Image, ImageDraw

import models
from common import TILE, ShrinkCrop
from convert import ANIME4K_SHRINK, ANIMEVIDEO_SHRINK


def synthetic_page(size=512, seed=0) -> Image.Image:
    """Line art, flat colour and a screentone patch, drawn at 4x and averaged down."""
    rng = np.random.default_rng(seed)
    s = 4
    img = Image.new("RGB", (size * s, size * s), (250, 248, 244))
    d = ImageDraw.Draw(img)
    d.polygon([(40 * s, 300 * s), (220 * s, 260 * s), (260 * s, 460 * s), (60 * s, 480 * s)], fill=(226, 120, 96))
    d.ellipse((300 * s, 40 * s, 470 * s, 210 * s), fill=(90, 130, 200))
    for y in range(280 * s, 500 * s, 7 * s):  # screentone dots
        for x in range(290 * s, 490 * s, 7 * s):
            d.ellipse((x, y, x + 3 * s, y + 3 * s), fill=(30, 30, 30))
    for _ in range(40):  # ink lines
        x0, y0 = rng.integers(0, size * s, 2)
        x1, y1 = x0 + rng.integers(-180 * s, 180 * s), y0 + rng.integers(-180 * s, 180 * s)
        d.line((x0, y0, x1, y1), fill=(15, 15, 15), width=int(rng.integers(2, 5)) * s // 2)
    return img.resize((size, size), Image.LANCZOS)


def to_tensor(img: Image.Image) -> torch.Tensor:
    arr = np.asarray(img.convert("RGB"), dtype=np.float32) / 255.0
    return torch.from_numpy(arr).permute(2, 0, 1).unsqueeze(0)


def to_image(t: torch.Tensor) -> Image.Image:
    arr = (t.squeeze(0).permute(1, 2, 0).clamp(0, 1).numpy() * 255.0).round().astype(np.uint8)
    return Image.fromarray(arr)


def psnr(a: torch.Tensor, b: torch.Tensor, border: int) -> float:
    a, b = a[..., border:-border, border:-border], b[..., border:-border, border:-border]
    mse = float(((a - b) ** 2).mean())
    return 99.0 if mse == 0 else 10 * math.log10(1.0 / mse)


def coreml_parity(package: str, torch_model: torch.nn.Module, failures: list):
    ml = ct.models.MLModel(package)
    tile = torch.from_numpy(np.random.default_rng(1).random((1, 3, TILE, TILE), dtype=np.float32))
    expected = torch_model(tile).detach().numpy()
    got = np.asarray(ml.predict({"input": tile.numpy()})["output"], dtype=np.float32)
    if got.shape != expected.shape:
        failures.append(f"{package}: shape {got.shape} != {expected.shape}")
        return
    diff = np.abs(got - expected)
    print(f"{os.path.basename(package)}: mean err {diff.mean() * 255:.3f}/255, max {diff.max() * 255:.2f}/255")
    if diff.mean() > 1.5 / 255 or np.percentile(diff, 99.9) > 8 / 255:
        failures.append(f"{package}: Core ML output differs from PyTorch")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--models", default="build/models")
    parser.add_argument("--work", default="build/work")
    parser.add_argument("--anime4k-variant", default="L")
    args = parser.parse_args()
    failures = []

    gt = synthetic_page()
    lr = gt.resize((gt.width // 2, gt.height // 2), Image.LANCZOS)
    x = to_tensor(lr)
    gt_t = to_tensor(gt)
    bicubic = to_tensor(lr.resize(gt.size, Image.BICUBIC))
    base = psnr(bicubic, gt_t, 32)
    print(f"bicubic PSNR {base:.2f} dB")

    animevideo = models.load_animevideo(os.path.join(args.work, "animevideov3.pth"))
    variant = args.anime4k_variant
    anime4k = models.load_anime4k(os.path.join(args.work, f"anime4k_{variant}.glsl"))

    results = {}
    with torch.no_grad():
        for name, net, slack in (("AnimeVideo-v3", animevideo, 4.0), ("Anime4K", anime4k, 1.0)):
            out = net(x)
            score = psnr(out, gt_t, 32)
            results[name] = out
            print(f"{name} PSNR {score:.2f} dB")
            # GAN-trained models can sit below bicubic on PSNR, but broken weights are far lower.
            if score < base - slack:
                failures.append(f"{name}: PSNR {score:.2f} dB is well below bicubic {base:.2f} dB")

        for package, net, shrink in (
            ("AnimeVideoV3_x2.mlpackage", animevideo, ANIMEVIDEO_SHRINK),
            (f"Anime4K_x2_{variant}.mlpackage", anime4k, ANIME4K_SHRINK),
        ):
            path = os.path.join(args.models, package)
            if os.path.exists(path):
                coreml_parity(path, ShrinkCrop(net, shrink), failures)
            else:
                failures.append(f"missing {path}")

    # side-by-side crop: ground truth | bicubic | AnimeVideo-v3 | Anime4K
    box = (240, 240, 400, 400)
    tiles = [gt, to_image(bicubic), to_image(results["AnimeVideo-v3"]), to_image(results["Anime4K"])]
    sheet = Image.new("RGB", (160 * 4 * 3, 160 * 3))
    for i, t in enumerate(tiles):
        sheet.paste(t.crop(box).resize((480, 480), Image.NEAREST), (i * 480, 0))
    os.makedirs(args.models, exist_ok=True)
    sheet.save(os.path.join(args.models, "compare.png"))
    to_image(results["AnimeVideo-v3"]).save(os.path.join(args.models, "sample_animevideo.png"))
    to_image(results["Anime4K"]).save(os.path.join(args.models, "sample_anime4k.png"))
    gt.save(os.path.join(args.models, "sample_original.png"))

    if failures:
        print("\nFAILED:")
        for f in failures:
            print(" -", f)
        sys.exit(1)
    print("\nall checks passed")


if __name__ == "__main__":
    main()
