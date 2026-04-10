#!/usr/bin/env python3
"""
Copyright (c) 2026 Nordic Semiconductor
SPDX-License-Identifier: LicenseRef-Nordic-5-Clause

Visualize int8 tensors from src/generated/test_images.h and compare to PC (TFLite) inputs.

Why 144 vs 146?
  The .tflite often reports the "logical" input (e.g. 144×144 NHWC). The Axon compiler may
  expose a larger *external* tensor (e.g. 146×146) when padding/striding is folded so the CPU
  fills one buffer the NPU expects. The compiler embeds the exact H×W in nrf_axon_model_*.h;
  firmware and test_images.h must match that header — not necessarily the raw .tflite tensor
  dims in Netron. Until you recompile so header == .tflite spatial size, PC and device inputs
  are different crops/resizes of the JPEG.

Usage (from applications/person_recognition):

  python3 scripts/visualize_compare_inputs.py \\
    --test-images-h src/generated/test_images.h \\
    --axon-header outputs/nrf_axon_model_mcunet_vww_320kb_.h \\
    --tflite models/mcunet-320kb-1mb_vww.tflite \\
    --pictures-dir pictures \\
    --out-dir debug_inputs
"""
from __future__ import annotations

import argparse
import importlib.util
import re
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
APP_DIR = SCRIPT_DIR.parent


def _load_embed_module():
    spec = importlib.util.spec_from_file_location(
        "embed_test_images", SCRIPT_DIR / "embed_test_images.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_test_images_header(h_path: Path) -> dict[str, np.ndarray]:
    """Parse all static const int8_t test_image_* arrays from generated header."""
    text = h_path.read_text(encoding="utf-8", errors="replace")
    tensors: dict[str, np.ndarray] = {}
    for m in re.finditer(
        r"static const int8_t (test_image_\w+)\[(\d+)\]\s*=\s*\{",
        text,
    ):
        name, n_s = m.group(1), int(m.group(2))
        start = m.end()
        end = text.find("};", start)
        if end < 0:
            raise ValueError(f"unclosed array {name}")
        body = text[start:end]
        nums = [int(x) for x in re.findall(r"-?\d+", body)]
        if len(nums) != n_s:
            raise ValueError(f"{name}: parsed {len(nums)} values, expected {n_s}")
        tensors[name] = np.array(nums, dtype=np.int8)
    if not tensors:
        raise ValueError(f"no test_image_* arrays in {h_path}")
    return tensors


def chw_int8_to_float_hwc(
    flat: np.ndarray,
    height: int,
    width: int,
    quant_mult: int,
    quant_round: int,
    quant_zp: int,
) -> np.ndarray:
    """Invert Axon input quant: float = (q - zp) * 2^round / mult. Shape (H,W,3)."""
    q = flat.reshape(3, height, width).astype(np.float64)
    scale = float(2**quant_round) / float(quant_mult)
    f = (q.astype(np.float64) - quant_zp) * scale
    return np.transpose(f, (1, 2, 0)).astype(np.float32)


def float_symmetric_hwc_to_rgb_u8(f_hwc: np.ndarray) -> np.ndarray:
    """Map [-1,1] float to uint8 RGB for display."""
    x01 = (f_hwc + 1.0) * 0.5
    return (np.clip(x01, 0.0, 1.0) * 255.0).astype(np.uint8)


def pc_tflite_float_hwc(
    image_path: Path, height: int, width: int
) -> np.ndarray:
    """Same float tensor as eval_det / run_tflite_reference before quant (H,W,3) in [-1,1]."""
    from PIL import Image

    im = Image.open(image_path).convert("RGB").resize((width, height), Image.Resampling.BILINEAR)
    x01 = np.array(im, dtype=np.float32) / 255.0
    return x01 * 2.0 - 1.0


def stem_to_test_var(stem: str) -> str:
    safe = stem.replace("-", "_")
    return "test_image_" + safe


def side_by_side(
    left: np.ndarray,
    right: np.ndarray,
    label_left: str,
    label_right: str,
) -> np.ndarray:
    from PIL import Image, ImageDraw, ImageFont

    h = max(left.shape[0], right.shape[0])
    w1, w2 = left.shape[1], right.shape[1]
    gap = 8
    banner = 28
    out = np.zeros((h + banner, w1 + gap + w2, 3), dtype=np.uint8)
    out[banner : banner + left.shape[0], :w1] = left
    out[banner : banner + right.shape[0], w1 + gap : w1 + gap + w2] = right
    pil = Image.fromarray(out)
    draw = ImageDraw.Draw(pil)
    try:
        font = ImageFont.truetype("DejaVuSans.ttf", 14)
    except OSError:
        font = ImageFont.load_default()
    draw.text((4, 4), label_left, fill=(255, 255, 0), font=font)
    draw.text((w1 + gap + 4, 4), label_right, fill=(0, 255, 255), font=font)
    return np.array(pil)


def report_diff_downscaled(
    device_f: np.ndarray,
    pc_f: np.ndarray,
) -> str:
    """Resize device map to pc shape with PIL and report max|diff| on [-1,1] float."""
    from PIL import Image

    dh, dw = device_f.shape[:2]
    ph, pw = pc_f.shape[:2]
    dev_u8 = float_symmetric_hwc_to_rgb_u8(device_f)
    im = Image.fromarray(dev_u8).resize((pw, ph), Image.Resampling.BILINEAR)
    dev_rs01 = np.array(im, dtype=np.float32) / 255.0
    dev_rs = dev_rs01 * 2.0 - 1.0
    d = np.abs(dev_rs - pc_f)
    return (
        f"  device {dh}x{dw} bilinear-resized to {ph}x{pw}: "
        f"max_abs_diff={float(d.max()):.4f} mean_abs_diff={float(d.mean()):.6f}\n"
        f"  (only meaningful if same semantic crop; different H×W usually dominates.)\n"
    )


def main() -> int:
    ap = argparse.ArgumentParser(description="Visualize test_images.h vs PC TFLite inputs")
    ap.add_argument(
        "--test-images-h",
        type=Path,
        default=APP_DIR / "src/generated/test_images.h",
    )
    ap.add_argument(
        "--axon-header",
        type=Path,
        default=APP_DIR / "outputs/nrf_axon_model_mcunet_vww_320kb_.h",
    )
    ap.add_argument(
        "--model-symbol",
        default="model_mcunet_vww_320kb",
    )
    ap.add_argument(
        "--tflite",
        type=Path,
        default=APP_DIR / "models/mcunet-320kb-1mb_vww.tflite",
    )
    ap.add_argument("--pictures-dir", type=Path, default=APP_DIR / "pictures")
    ap.add_argument("--out-dir", type=Path, default=APP_DIR / "debug_inputs")
    ap.add_argument(
        "--jpeg-stems",
        nargs="+",
        default=["demo_picture", "demo_2", "demo_3"],
        help="Base names matching pictures/<stem>.jpeg",
    )
    args = ap.parse_args()

    if not args.test_images_h.is_file():
        print(f"Missing {args.test_images_h}", file=sys.stderr)
        return 1
    if not args.axon_header.is_file():
        print(f"Missing {args.axon_header}", file=sys.stderr)
        return 1

    embed = _load_embed_module()
    spec = embed.parse_axon_input_from_header(args.axon_header, args.model_symbol)
    ah, aw = spec["height"], spec["width"]
    qm, qr, qz = spec["quant_mult"], spec["quant_round"], spec["quant_zp"]

    tensors = parse_test_images_header(args.test_images_h)

    tf_hw = None
    if args.tflite.is_file():
        try:
            import tensorflow as tf

            interp = tf.lite.Interpreter(model_path=str(args.tflite))
            interp.allocate_tensors()
            shp = interp.get_input_details()[0]["shape"]
            _, tf_hw = int(shp[1]), int(shp[2])
        except Exception as e:
            print(f"Warning: could not read TFLite shape: {e}", file=sys.stderr)

    args.out_dir.mkdir(parents=True, exist_ok=True)
    lines = [
        "Input geometry report\n",
        f"  Axon header (firmware / test_images.h): {ah} x {aw}\n",
    ]
    if tf_hw is not None:
        lines.append(f"  TFLite file input tensor: {tf_hw} x {tf_hw}\n")
        if tf_hw != ah:
            lines.append(
                "  *** MISMATCH: PC interpreter and device use different H×W; "
                "images are not comparable pixel-for-pixel.\n"
            )
    lines.append(
        "\nPadding note: if the network uses internal padding, the compiler may still "
        "expose one external buffer size in the .h. That size is what you must feed; "
        "it need not equal the .tflite spatial dims until graphs are aligned.\n\n"
    )

    from PIL import Image

    for stem in args.jpeg_stems:
        jpeg = args.pictures_dir / f"{stem}.jpeg"
        if not jpeg.is_file():
            jpeg = args.pictures_dir / f"{stem}.jpg"
        var = stem_to_test_var(stem)
        if var not in tensors:
            print(f"skip {stem}: no {var} in header", file=sys.stderr)
            continue
        flat = tensors[var]
        expected = 3 * ah * aw
        if flat.size != expected:
            print(
                f"skip {stem}: {var} len {flat.size} != 3*{ah}*{aw}={expected}",
                file=sys.stderr,
            )
            continue

        f_dev = chw_int8_to_float_hwc(flat, ah, aw, qm, qr, qz)
        rgb_dev = float_symmetric_hwc_to_rgb_u8(f_dev)
        Image.fromarray(rgb_dev).save(args.out_dir / f"{stem}_device_from_header.png")

        # Sanity: regenerate with embed pipeline and compare to parsed C array
        regen = embed.load_and_quantize(
            jpeg, ah, aw, spec["use_chw"], qm, qr, qz, "symmetric_m1_1"
        )
        if not np.array_equal(regen.flatten(), flat):
            lines.append(f"WARNING {stem}: regenerated int8 != parsed header (stale build?)\n")
        else:
            lines.append(f"OK {stem}: parsed header matches embed_test_images.py regeneration\n")

        if tf_hw is not None and jpeg.is_file():
            f_pc = pc_tflite_float_hwc(jpeg, tf_hw, tf_hw)
            rgb_pc = float_symmetric_hwc_to_rgb_u8(f_pc)
            Image.fromarray(rgb_pc).save(
                args.out_dir / f"{stem}_pc_tflite_{tf_hw}x{tf_hw}.png"
            )
            combo = side_by_side(
                rgb_dev,
                rgb_pc,
                f"device header {ah}x{aw}",
                f"PC TFLite {tf_hw}x{tf_hw}",
            )
            Image.fromarray(combo).save(args.out_dir / f"{stem}_side_by_side.png")
            lines.append(report_diff_downscaled(f_dev, f_pc))

    report_path = args.out_dir / "compare_report.txt"
    report_path.write_text("".join(lines), encoding="utf-8")
    print("".join(lines))
    print(f"Wrote PNGs and {report_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
