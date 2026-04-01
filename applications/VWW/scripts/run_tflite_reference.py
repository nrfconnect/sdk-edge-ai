#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Run MCUNet VWW TFLite on disk the same way as eval_det.py (resize, [-1,1], quantize, invoke).
Use to compare serial log from the board against PC ground truth for the same JPEGs.

  python3 scripts/run_tflite_reference.py \\
    --tflite models/mcunet-320kb-1mb_vww.tflite \\
    pictures/demo_picture.jpeg pictures/demo_2.jpeg
"""
import argparse
import sys
from pathlib import Path

import numpy as np


def quantize_input(float_nhwc: np.ndarray, input_detail: dict) -> np.ndarray:
    d = input_detail
    dtype = d.get("dtype")
    if dtype not in (np.int8, np.uint8):
        return float_nhwc.astype(np.float32)
    q = d.get("quantization_parameters") or {}
    scales, zps = q.get("scales"), q.get("zero_points")
    if not scales or not zps:
        return float_nhwc.astype(np.float32)
    scale = float(scales[0])
    zp = int(zps[0])
    quantized = np.round(float_nhwc / scale).astype(np.int32) + zp
    if dtype == np.int8:
        return np.clip(quantized, -128, 127).astype(np.int8)
    return np.clip(quantized, 0, 255).astype(np.uint8)


def dequantize_output(out: np.ndarray, out_detail: dict) -> np.ndarray:
    if out.dtype not in (np.int8, np.uint8):
        return out.astype(np.float32)
    q = out_detail.get("quantization_parameters") or {}
    scales, zps = q.get("scales"), q.get("zero_points")
    if scales is None or zps is None:
        return out.astype(np.float32)
    scale = float(scales[0])
    zp = int(zps[0])
    return (out.astype(np.float32) - zp) * scale


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tflite", default="models/mcunet-320kb-1mb_vww.tflite")
    ap.add_argument("images", nargs="+", help="JPEG paths")
    args = ap.parse_args()
    root = Path(__file__).resolve().parent.parent
    tflite_path = root / args.tflite
    if not tflite_path.is_file():
        print(f"Missing {tflite_path}", file=sys.stderr)
        return 1
    try:
        import tensorflow as tf
    except ImportError:
        print("pip install tensorflow", file=sys.stderr)
        return 1

    interp = tf.lite.Interpreter(model_path=str(tflite_path))
    interp.allocate_tensors()
    inp_d = interp.get_input_details()[0]
    out_d = interp.get_output_details()[0]
    shp = inp_d["shape"]
    _, h, w, _ = [int(x) for x in shp]

    print("TFLite input", shp, "quantization", inp_d.get("quantization"))
    print("TFLite output", out_d["shape"], "quantization", out_d.get("quantization"))
    print()

    for img_path in args.images:
        p = Path(img_path)
        if not p.is_file():
            p = root / "pictures" / img_path
        if not p.is_file():
            print(f"skip missing {img_path}", file=sys.stderr)
            continue
        from PIL import Image

        im = Image.open(p).convert("RGB").resize((w, h))
        x01 = np.array(im, dtype=np.float32) / 255.0
        x = (x01 * 2.0 - 1.0)[np.newaxis, ...]
        q = quantize_input(x, inp_d)
        interp.set_tensor(inp_d["index"], q)
        interp.invoke()
        raw = interp.get_tensor(out_d["index"])
        logits = dequantize_output(np.squeeze(raw), out_d)
        logits = np.asarray(logits, dtype=np.float64).reshape(-1)
        m = logits.max()
        e = np.exp(logits - m)
        prob = e / e.sum()
        print(f"{p.name}: logits {logits}  P(person)={prob[1]:.4f}  argmax={int(np.argmax(logits))}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
