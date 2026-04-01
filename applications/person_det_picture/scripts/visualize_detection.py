#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Draw person_det bounding box (x1,y1,x2,y2 in model pixel space, width x height)
on either a normal image (resized to model size) or on a decoded view of int8 CHW
tensor (same layout as model input / embed_demo_input.py).
"""
from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as e:
    raise SystemExit("Pillow required: pip install Pillow") from e


def parse_axon_input_from_header(header_path: Path, model_symbol: str) -> dict:
    text = header_path.read_text(encoding="utf-8", errors="replace")
    anchor = f"const nrf_axon_nn_compiled_model_s {model_symbol}"
    idx = text.find(anchor)
    if idx < 0:
        raise ValueError(f"Could not find '{anchor}' in {header_path}")
    chunk = text[idx : idx + 12000]
    dim_m = re.search(
        r"\.dimensions\s*=\s*\{\s*"
        r"\.height\s*=\s*(\d+),\s*"
        r"\.width\s*=\s*(\d+),\s*"
        r"\.channel_cnt\s*=\s*(\d+),",
        chunk,
        re.DOTALL,
    )
    if not dim_m:
        raise ValueError(f"Could not parse .dimensions in {header_path}")
    height, width, chans = (int(dim_m.group(1)), int(dim_m.group(2)), int(dim_m.group(3)))
    qm_m = re.search(r"\.quant_mult\s*=\s*(\d+),", chunk)
    qr_m = re.search(r"\.quant_round\s*=\s*(\d+),", chunk)
    zp_m = re.search(r"\.quant_zp\s*=\s*(-?\d+),", chunk)
    st_m = re.search(r"\.stride\s*=\s*(\d+),", chunk)
    if not all([qm_m, qr_m, zp_m, st_m]):
        raise ValueError(f"Could not parse quant/stride in {header_path}")
    stride = int(st_m.group(1))
    if stride == width:
        use_chw = True
    elif stride == width * chans:
        use_chw = False
    else:
        use_chw = True
    return {
        "height": height,
        "width": width,
        "channel_cnt": chans,
        "quant_mult": int(qm_m.group(1)),
        "quant_round": int(qr_m.group(1)),
        "quant_zp": int(zp_m.group(1)),
        "use_chw": use_chw,
    }


def dequant_chw_int8_to_rgb_u8(
    q: np.ndarray,
    quant_mult: int,
    quant_round: int,
    quant_zp: int,
) -> np.ndarray:
    """
    Inverse of embed_demo_input: int8 CHW -> approximate float in [-1,1] -> uint8 HWC.
    """
    if q.dtype != np.int8 and q.dtype != np.int16:
        q = q.astype(np.int32)
    qf = q.astype(np.float64)
    # x = ((q - zp) * 2**round) / mult  (matches integer quant in embed)
    x = ((qf - quant_zp) * (2**quant_round)) / float(quant_mult)
    x = np.clip(x, -1.0, 1.0)
    u8 = np.round((x + 1.0) * 0.5 * 255.0).astype(np.uint8)
    if u8.ndim != 3:
        raise ValueError(f"expected CHW tensor, got shape {u8.shape}")
    # CHW -> HWC
    return np.transpose(u8, (1, 2, 0))


def load_tensor_npy(path: Path) -> np.ndarray:
    arr = np.load(path)
    if arr.dtype != np.int8:
        arr = arr.astype(np.int8)
    return arr


def parse_box(s: str) -> tuple[float, float, float, float]:
    parts = [p.strip() for p in s.replace(" ", "").split(",")]
    if len(parts) != 4:
        raise ValueError("box must be four numbers: x1,y1,x2,y2")
    return tuple(float(p) for p in parts)


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Visualize person_det box on an image or int8 CHW tensor (model input)."
    )
    ap.add_argument(
        "--box",
        required=True,
        help="Bounding box in model coordinates: x1,y1,x2,y2 (e.g. 59.5,52.8,153.0,125.4)",
    )
    ap.add_argument("--score", type=float, default=None, help="Optional score label")
    ap.add_argument(
        "--axon-header",
        type=Path,
        default=None,
        help="nrf_axon_model_person_det_.h (default: ../person_recognition/outputs/...)",
    )
    ap.add_argument("--model-symbol", default="model_person_det")
    ap.add_argument(
        "--image",
        type=Path,
        default=None,
        help="RGB image (JPEG/PNG); resized to model WxH for drawing",
    )
    ap.add_argument(
        "--tensor-npy",
        type=Path,
        default=None,
        help="NumPy file int8 CHW shape (3, H, W), same as model input",
    )
    ap.add_argument(
        "--out",
        type=Path,
        default=Path("detection_visualization.png"),
        help="Output PNG path",
    )
    ap.add_argument("--line-width", type=int, default=2)
    args = ap.parse_args()

    root = Path(__file__).resolve().parent.parent
    hdr = args.axon_header
    if hdr is None:
        hdr = root.parent / "person_recognition" / "outputs" / "nrf_axon_model_person_det_.h"
    if not hdr.is_file():
        print(f"Error: axon header not found: {hdr}", file=sys.stderr)
        return 1

    try:
        spec = parse_axon_input_from_header(hdr, args.model_symbol)
    except ValueError as e:
        print(f"Error: {e}", file=sys.stderr)
        return 1

    w, h = spec["width"], spec["height"]
    x1, y1, x2, y2 = parse_box(args.box)

    if args.tensor_npy is not None and args.image is not None:
        print("Using --tensor-npy for background (--image ignored).", file=sys.stderr)

    if args.tensor_npy is not None:
        chw = load_tensor_npy(args.tensor_npy)
        if chw.shape != (spec["channel_cnt"], h, w):
            print(
                f"Error: tensor shape {chw.shape}, expected ({spec['channel_cnt']}, {h}, {w})",
                file=sys.stderr,
            )
            return 1
        rgb = dequant_chw_int8_to_rgb_u8(
            chw, spec["quant_mult"], spec["quant_round"], spec["quant_zp"]
        )
        pil = Image.fromarray(rgb)
    elif args.image is not None:
        if not args.image.is_file():
            print(f"Error: image not found: {args.image}", file=sys.stderr)
            return 1
        img = Image.open(args.image).convert("RGB")
        img = img.resize((w, h), Image.Resampling.BILINEAR)
        pil = img
    else:
        print("Error: provide --tensor-npy and/or --image (at least one).", file=sys.stderr)
        return 1

    dr = ImageDraw.Draw(pil)
    xy = [x1, y1, x2, y2]
    # PIL needs integer pixel coords for outline
    box_i = [int(round(c)) for c in xy]
    dr.rectangle(box_i, outline="red", width=args.line_width)

    label = None
    if args.score is not None:
        label = f"score {args.score:.3f}"
    if label:
        try:
            font = ImageFont.load_default()
        except Exception:
            font = None
        tx, ty = box_i[0], max(0, box_i[1] - 12)
        dr.text((tx, ty), label, fill="red", font=font)

    args.out.parent.mkdir(parents=True, exist_ok=True)
    pil.save(args.out)
    print(f"Wrote {args.out} ({w}x{h} model space, box {xy})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
