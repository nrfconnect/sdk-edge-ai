#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Prepares demo_picture.jpeg as a single test vector for the person-det compiler.
Output shape matches the TFLite model input (1, H, W, C). Data is float32;
the Axon compiler will quantize when building.
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def get_model_input_shape(tflite_path: str) -> tuple:
    try:
        import tensorflow as tf
    except ImportError:
        try:
            import tflite
            interp = tflite.Interpreter(model_path=tflite_path)
            interp.allocate_tensors()
            inp = interp.get_input_details()[0]
            return tuple(inp["shape"])
        except Exception as e:
            raise RuntimeError(
                "Need tensorflow or tflite_runtime. Install with: pip install tensorflow"
            ) from e
    interp = tf.lite.Interpreter(model_path=tflite_path)
    interp.allocate_tensors()
    inp = interp.get_input_details()[0]
    return tuple(inp["shape"])


def load_and_resize_image(image_path: str, target_shape: tuple) -> np.ndarray:
    """Load image, resize to (H, W), ensure 3 channels, normalize to [0, 1]."""
    try:
        from PIL import Image
    except ImportError:
        raise RuntimeError("PIL required. Install with: pip install Pillow")
    img = Image.open(image_path)
    if img.mode != "RGB":
        img = img.convert("RGB")
    img = np.array(img, dtype=np.float32) / 255.0
    # target_shape is (1, H, W, C)
    _, h, w, c = target_shape
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
        default="pictures/demo_picture.jpeg",
        help="Path to input image (default: pictures/demo_picture.jpeg)",
    )
    ap.add_argument(
        "--out",
        default="data/demo_input.npy",
        help="Output .npy path (default: data/demo_input.npy)",
    )
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    model_path = root / args.model
    image_path = root / args.image
    out_path = root / args.out
    if not model_path.is_file():
        print(f"Error: model not found: {model_path}", file=sys.stderr)
        return 1
    if not image_path.is_file():
        print(f"Error: image not found: {image_path}", file=sys.stderr)
        return 1
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shape = get_model_input_shape(str(model_path))
    arr = load_and_resize_image(str(image_path), shape)
    np.save(str(out_path), arr)
    print(f"Saved shape {arr.shape} to {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
