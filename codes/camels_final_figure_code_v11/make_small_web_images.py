#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Compress paper figures from All_plots into small web-friendly images.

What it does
------------
- Reads image files from an All_plots folder.
- Skips All_plots/small and All_plots/others.
- Saves compressed web-friendly copies into All_plots/small.
- Default output format is WebP because it usually gives the best quality/size ratio.
- Strips metadata and uses optimized saving.
- Keeps a reasonable visual quality for quick web/email/ChatGPT review.

Usage
-----
Option 1: run from the project root:
    python make_small_web_images.py

Option 2: give the All_plots folder explicitly:
    python make_small_web_images.py --all-plots "F:/Experiments/CAMELS_US/Clean_4_upload/results/All_plots"

Useful options:
    python make_small_web_images.py --max-width 1800 --quality 72
    python make_small_web_images.py --format jpg --quality 78
    python make_small_web_images.py --also-jpg

Dependencies
------------
    pip install pillow
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Iterable

from PIL import Image, ImageOps


IMAGE_EXTS = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create small web-friendly versions of images in results/All_plots."
    )
    parser.add_argument(
        "--all-plots",
        type=str,
        default=None,
        help=(
            "Path to All_plots. If omitted, the script uses "
            "<current working directory>/results/All_plots."
        ),
    )
    parser.add_argument(
        "--out-subdir",
        type=str,
        default="small",
        help="Output subfolder inside All_plots. Default: small",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="webp",
        choices=["webp", "jpg", "png"],
        help="Primary output format. Default: webp",
    )
    parser.add_argument(
        "--also-jpg",
        action="store_true",
        help="Also save a JPG copy in addition to the primary format.",
    )
    parser.add_argument(
        "--max-width",
        type=int,
        default=1800,
        help="Maximum output width in pixels. Default: 1800",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=1800,
        help="Maximum output height in pixels. Default: 1800",
    )
    parser.add_argument(
        "--quality",
        type=int,
        default=72,
        help="Compression quality for WebP/JPG, 1-100. Default: 72",
    )
    parser.add_argument(
        "--png-compress-level",
        type=int,
        default=9,
        help="PNG compression level, 0-9. Default: 9",
    )
    parser.add_argument(
        "--include-pdf",
        action="store_true",
        help="Ignored by default; this script compresses raster images only.",
    )
    return parser.parse_args()


def find_all_plots(path_arg: str | None) -> Path:
    if path_arg:
        all_plots = Path(path_arg).expanduser().resolve()
    else:
        all_plots = (Path.cwd() / "results" / "All_plots").resolve()

    if not all_plots.exists():
        raise FileNotFoundError(f"All_plots folder not found: {all_plots}")

    if not all_plots.is_dir():
        raise NotADirectoryError(f"Not a folder: {all_plots}")

    return all_plots


def iter_images(all_plots: Path, out_dir: Path) -> Iterable[Path]:
    for p in sorted(all_plots.iterdir()):
        if not p.is_file():
            continue
        if p.parent == out_dir:
            continue
        if p.suffix.lower() not in IMAGE_EXTS:
            continue
        yield p


def resize_image(img: Image.Image, max_width: int, max_height: int) -> Image.Image:
    img = ImageOps.exif_transpose(img)

    # Convert transparent images onto white background for JPG compatibility.
    # WebP can keep alpha, but paper plots normally do not need transparency.
    if img.mode not in ("RGB", "RGBA"):
        img = img.convert("RGBA" if "A" in img.getbands() else "RGB")

    w, h = img.size
    scale = min(max_width / w, max_height / h, 1.0)
    if scale < 1.0:
        new_size = (max(1, int(w * scale)), max(1, int(h * scale)))
        img = img.resize(new_size, Image.Resampling.LANCZOS)

    return img


def flatten_to_white(img: Image.Image) -> Image.Image:
    """Return an RGB image with transparency flattened onto white."""
    if img.mode == "RGBA":
        bg = Image.new("RGB", img.size, "white")
        bg.paste(img, mask=img.getchannel("A"))
        return bg
    if img.mode != "RGB":
        return img.convert("RGB")
    return img


def save_webp(img: Image.Image, out_path: Path, quality: int) -> None:
    # WebP with method=6 is slower but gives smaller files.
    img.save(
        out_path,
        "WEBP",
        quality=quality,
        method=6,
        optimize=True,
        lossless=False,
    )


def save_jpg(img: Image.Image, out_path: Path, quality: int) -> None:
    rgb = flatten_to_white(img)
    rgb.save(
        out_path,
        "JPEG",
        quality=quality,
        optimize=True,
        progressive=True,
        subsampling="4:2:0",
    )


def save_png(img: Image.Image, out_path: Path, compress_level: int) -> None:
    # PNG is lossless, so files may remain much larger than WebP/JPG.
    img.save(
        out_path,
        "PNG",
        optimize=True,
        compress_level=compress_level,
    )


def human_size(num_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    n = float(num_bytes)
    for u in units:
        if n < 1024 or u == units[-1]:
            return f"{n:.1f} {u}"
        n /= 1024
    return f"{num_bytes} B"


def main() -> None:
    args = parse_args()
    all_plots = find_all_plots(args.all_plots)
    out_dir = all_plots / args.out_subdir
    out_dir.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(all_plots, out_dir))
    if not images:
        print(f"No image files found in: {all_plots}")
        return

    total_in = 0
    total_out = 0
    saved_count = 0

    print(f"Input folder : {all_plots}")
    print(f"Output folder: {out_dir}")
    print(f"Images found : {len(images)}")
    print("-" * 80)

    for src in images:
        try:
            with Image.open(src) as im:
                img = resize_image(im, args.max_width, args.max_height)

                original_size = src.stat().st_size
                total_in += original_size

                outputs = []

                if args.format == "webp":
                    out = out_dir / f"{src.stem}.webp"
                    save_webp(img, out, args.quality)
                    outputs.append(out)

                elif args.format == "jpg":
                    out = out_dir / f"{src.stem}.jpg"
                    save_jpg(img, out, args.quality)
                    outputs.append(out)

                elif args.format == "png":
                    out = out_dir / f"{src.stem}.png"
                    save_png(img, out, args.png_compress_level)
                    outputs.append(out)

                if args.also_jpg and args.format != "jpg":
                    out_jpg = out_dir / f"{src.stem}.jpg"
                    save_jpg(img, out_jpg, args.quality)
                    outputs.append(out_jpg)

                out_size = sum(p.stat().st_size for p in outputs if p.exists())
                total_out += out_size
                saved_count += len(outputs)

                ratio = 100.0 * out_size / original_size if original_size else 0.0
                output_names = ", ".join(p.name for p in outputs)
                print(
                    f"{src.name:48s} -> {output_names:48s} "
                    f"{human_size(original_size):>9s} -> {human_size(out_size):>9s} "
                    f"({ratio:5.1f}%)"
                )

        except Exception as e:
            print(f"[WARN] Skipped {src.name}: {e}")

    print("-" * 80)
    print(f"Saved files : {saved_count}")
    print(f"Total input : {human_size(total_in)}")
    print(f"Total output: {human_size(total_out)}")
    if total_in:
        print(f"Final ratio : {100.0 * total_out / total_in:.1f}% of original size")
    print("Done.")


if __name__ == "__main__":
    main()
