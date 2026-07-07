#!/usr/bin/env python3
"""Resize ArabicSign images to width 340.

The height is scaled proportionally, so images are not distorted.

Default input:
    data/ArabicSign

Default output:
    data/ArabicSign_lowres

The output keeps the same directory structure as the input.
"""

import json
import os
from pathlib import Path

from PIL import Image, ImageFile
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
CORRUPTED_LOG = SCRIPT_DIR / "arabicsign_corrupted_images.json"

# Allow loading truncated/corrupted images when possible.
ImageFile.LOAD_TRUNCATED_IMAGES = True

SRC_ROOT = SCRIPT_DIR / "ArabicSign"
DST_ROOT = SCRIPT_DIR / "ArabicSign_lowres"
TARGET_WIDTH = 340


def resize_image(src_path: str, dst_path: str) -> None:
    """Resize image to target_width, scaling height to preserve aspect ratio."""
    with Image.open(src_path) as img:
        img.load()
        w, h = img.size
        if w == TARGET_WIDTH:
            new_h = h
        else:
            scale = TARGET_WIDTH / w
            new_h = int(round(h * scale))
        new_size = (TARGET_WIDTH, new_h)
        resized = img.resize(new_size, Image.Resampling.LANCZOS)
        resized.save(dst_path, quality=95)


def main() -> None:
    src = Path(SRC_ROOT)
    dst = Path(DST_ROOT)
    if not src.exists():
        raise FileNotFoundError(f"Source directory not found: {src}")

    image_exts = {".jpg", ".jpeg", ".png", ".bmp", ".gif", ".tiff", ".webp"}
    to_process = []
    for root, _, files in os.walk(src):
        root_p = Path(root)
        rel_root = root_p.relative_to(src)
        for f in files:
            if Path(f).suffix.lower() in image_exts:
                src_file = root_p / f
                dst_file = dst / rel_root / f
                to_process.append((str(src_file), str(dst_file)))

    print(f"Found {len(to_process)} images to resize (width={TARGET_WIDTH})")
    skipped = []
    for src_path, dst_path in tqdm(to_process, desc="Resizing"):
        os.makedirs(os.path.dirname(dst_path), exist_ok=True)
        try:
            resize_image(src_path, dst_path)
        except OSError as e:
            rel_path = str(Path(src_path).relative_to(src))
            skipped.append({"path": rel_path, "full_path": src_path, "error": str(e)})
    if skipped:
        with open(CORRUPTED_LOG, "w") as f:
            json.dump(skipped, f, indent=2)
        print(f"\nSkipped {len(skipped)} images (recorded to {CORRUPTED_LOG}):")
        for item in skipped[:10]:
            print(f"  {item['path']}: {item['error']}")
        if len(skipped) > 10:
            print(f"  ... and {len(skipped) - 10} more")
    else:
        if CORRUPTED_LOG.exists():
            CORRUPTED_LOG.unlink()
        print("\nNo corrupted images encountered.")

    print(f"Done. Low-res dataset saved to {DST_ROOT}")


if __name__ == "__main__":
    main()
