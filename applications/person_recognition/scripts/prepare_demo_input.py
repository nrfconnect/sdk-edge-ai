#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Prepares a single test vector for the person-det compiler.
The output tensor is float32 with shape (1, 128, 160, 3), matching
person-det.tflite expected NHWC input.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

TARGET_SHAPE = (1, 128, 160, 3)
SUPPORTED_EXTENSIONS = (".jpg", ".jpeg", ".png")

def resolve_image_path(root: Path, image_arg: str, pictures_dir: str) -> Path:
    pictures_path = root / pictures_dir
    if not pictures_path.is_dir():
        raise RuntimeError(f"Pictures directory not found: {pictures_path}")

    if image_arg:
        image_path = root / image_arg
        if not image_path.is_file():
            raise RuntimeError(f"Image not found: {image_path}")
        return image_path

    candidates = sorted(
        p for p in pictures_path.iterdir() if p.is_file() and p.suffix.lower() in SUPPORTED_EXTENSIONS
    )
    if not candidates:
        raise RuntimeError(
            f"No supported images found in {pictures_path}. "
            f"Expected one of: {', '.join(SUPPORTED_EXTENSIONS)}"
        )
    return candidates[0]


def load_and_resize_image(image_path: str) -> np.ndarray:
    """Load image, resize to (H, W), ensure 3 channels, normalize to [0, 1]."""
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("PIL required. Install with: pip install Pillow")
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = np.array(img, dtype=np.float32) / 255.0
    _, h, w, c = TARGET_SHAPE
    if img.shape[0] != h or img.shape[1] != w:
        from PIL import Image as PILImage
        pil_img = PILImage.fromarray((img * 255).astype(np.uint8))
        pil_img = pil_img.resize((w, h), PILImage.Resampling.BILINEAR)
        img = np.array(pil_img, dtype=np.float32) / 255.0
    img = np.expand_dims(img, axis=0)
    return img.astype(np.float32)


def main() -> int:
    ap = argparse.ArgumentParser(description="Prepare demo image as .npy for Axon compiler")
    ap.add_argument(
        "--model",
        default="models/person-det.tflite",
        help="Path to person-det.tflite (default: models/person-det.tflite)",
    )
    ap.add_argument(
        "--image",
        default="",
        help=(
            "Path to one input image. If omitted, first supported image in "
            "pictures directory is used."
        ),
    )
    ap.add_argument(
        "--pictures-dir",
        default="pictures",
        help="Path to pictures directory (default: pictures)",
    )
    ap.add_argument(
        "--out",
        default="data/demo_input.npy",
        help="Output .npy path (default: data/demo_input.npy)",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    model_path = root / args.model
    out_path = root / args.out
    if not model_path.is_file():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        return 1

    try:
        image_path = resolve_image_path(root, args.image, args.pictures_dir)
    except RuntimeError as err:
        print(f"Error: {err}", file=sys.stderr)
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)
    arr = load_and_resize_image(str(image_path))
    if arr.shape != TARGET_SHAPE:
        print(
            f"Error: generated tensor shape {arr.shape}, expected {TARGET_SHAPE}",
            file=sys.stderr,
        )
        return 1
    np.save(str(out_path), arr)
    print(f"Using image: {image_path}")
    print(f"Saved shape {arr.shape}, dtype {arr.dtype} to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
