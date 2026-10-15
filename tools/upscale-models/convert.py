"""Converts the bundled upscalers to Core ML `.mlpackage` files (run on macOS).

    python convert.py --out build/models [--anime4k-variant L]
"""

import argparse
import os

import models
from common import ShrinkCrop, convert

ANIMEVIDEO_SHRINK = 16
ANIME4K_SHRINK = 8


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="build/models")
    parser.add_argument("--work", default="build/work")
    parser.add_argument("--anime4k-variant", default="L", choices=["S", "M", "L", "VL", "UL"])
    args = parser.parse_args()
    os.makedirs(args.work, exist_ok=True)

    weights = models.download(models.ANIMEVIDEO_URL, os.path.join(args.work, "animevideov3.pth"))
    animevideo = ShrinkCrop(models.load_animevideo(weights), ANIMEVIDEO_SHRINK)
    print("wrote", convert(animevideo, "AnimeVideoV3_x2", args.out))

    variant = args.anime4k_variant
    shader = models.download(
        models.ANIME4K_URL.format(variant=variant),
        os.path.join(args.work, f"anime4k_{variant}.glsl"),
    )
    anime4k = ShrinkCrop(models.load_anime4k(shader), ANIME4K_SHRINK)
    print("wrote", convert(anime4k, f"Anime4K_x2_{variant}", args.out))


if __name__ == "__main__":
    main()
